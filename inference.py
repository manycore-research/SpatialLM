import os
import glob
import argparse

import torch
import numpy as np
import open3d as o3d
from tqdm import tqdm
from threading import Thread
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TextIteratorStreamer, set_seed

from spatiallm import Layout
from spatiallm.pcd import load_o3d_pcd, get_points_and_colors, cleanup_pcd, Compose

DETECT_TYPE_PROMPT = {
    "all": "Detect walls, doors, windows, boxes.",
    "arch": "Detect walls, doors, windows.",
    "object": "Detect boxes.",
}


def save_processed_pointcloud(point_cloud_tensor, output_path, original_min_extent=None):
    """
    Save the processed point cloud tensor as a PLY file to visualize what the model sees.
    
    Args:
        point_cloud_tensor: The processed point cloud tensor from preprocess_point_cloud
        output_path: Path to save the PLY file
        original_min_extent: Original minimum extent for coordinate restoration
    """
    # Extract the point cloud data (remove batch dimension)
    point_data = point_cloud_tensor.squeeze(0).cpu().numpy()
    
    # The tensor structure is [grid_coord(3), xyz(3), rgb(3)] = 9 columns
    if point_data.shape[1] >= 9:
        grid_coords = point_data[:, 0:3]  # Discretized grid coordinates
        xyz_coords = point_data[:, 3:6]   # Actual XYZ coordinates (after PositiveShift)
        rgb_colors = point_data[:, 6:9]   # RGB colors (normalized)
        
        # Use actual XYZ coordinates (these are the shifted/processed coordinates)
        points = xyz_coords
        colors = rgb_colors
        
        # Restore original coordinate system if min_extent provided
        if original_min_extent is not None:
            points = points + original_min_extent
        
        # Ensure colors are in [0,1] range
        colors = np.clip(colors, 0, 1)
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Save as PLY
        o3d.io.write_point_cloud(output_path, pcd)
        print(f"Saved processed point cloud to: {output_path}")
        print(f"  - Number of points: {len(points)}")
        print(f"  - Coordinate range: [{points.min(axis=0)}, {points.max(axis=0)}]")
        print(f"  - Grid coordinate range: [{grid_coords.min(axis=0)}, {grid_coords.max(axis=0)}]")
    else:
        print(f"Warning: Unexpected tensor shape {point_data.shape}, cannot save PLY")


def save_grid_coordinates_pointcloud(point_cloud_tensor, output_path, grid_size_m=None, min_extent=None):
    """
    Save a point cloud using the discretized grid coordinates to see the voxel grid structure.
    """
    point_data = point_cloud_tensor.squeeze(0).cpu().numpy()
    
    if point_data.shape[1] >= 9:
        grid_coords = point_data[:, 0:3]  # Discretized grid coordinates
        rgb_colors = point_data[:, 6:9]   # RGB colors
        
        # Use grid coordinates as positions (voxel grid). Optionally convert to meters for downstream use
        if grid_size_m is not None:
            points = grid_coords.astype(float) * float(grid_size_m)
            if min_extent is not None:
                points = points + np.array(min_extent, dtype=float)
        else:
            points = grid_coords
        colors = np.clip(rgb_colors, 0, 1)
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(float))
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Save as PLY
        o3d.io.write_point_cloud(output_path, pcd)
        print(f"Saved grid coordinate point cloud to: {output_path}")
        print(f"  - Number of voxels: {len(points)}")
        print(f"  - Grid coordinate range: [{points.min(axis=0)}, {points.max(axis=0)}]")


def preprocess_point_cloud_adaptive_unified(points, colors, base_grid_size, num_bins, density_levels=3):
    """
    Unified adaptive preprocessing that maintains single coordinate system.
    Uses density-weighted sampling within a single grid.
    """
    from sklearn.neighbors import NearestNeighbors
    
    print(f"Unified adaptive preprocessing with {density_levels} density levels...")
    
    # Calculate local point density
    k_neighbors = min(20, len(points) // 10)
    nbrs = NearestNeighbors(n_neighbors=k_neighbors).fit(points)
    distances, _ = nbrs.kneighbors(points)
    local_density = 1.0 / (distances[:, -1] + 1e-6)
    
    # Normalize density to [0, 1] range
    min_density = np.min(local_density)
    max_density = np.max(local_density)
    normalized_density = (local_density - min_density) / (max_density - min_density + 1e-6)
    
    # Create density-based sampling weights
    # Higher density points get higher sampling probability
    sampling_weights = 0.1 + 0.9 * normalized_density  # Range [0.1, 1.0]
    
    # Use adaptive grid size based on overall density distribution
    density_percentile_75 = np.percentile(normalized_density, 75)
    adaptive_grid_scale = 1.0 / (1.0 + density_percentile_75)  # Smaller grid for denser clouds
    final_grid_size = base_grid_size * adaptive_grid_scale
    
    print(f"  Adaptive grid size: {final_grid_size:.6f} (scale: {adaptive_grid_scale:.3f})")
    print(f"  Density range: [{min_density:.2f}, {max_density:.2f}]")
    print(f"  Sampling weights range: [{np.min(sampling_weights):.2f}, {np.max(sampling_weights):.2f}]")
    
    # Apply standard preprocessing with adaptive grid
    transform = Compose([
        dict(type="PositiveShift"),
        dict(type="NormalizeColor"),
        dict(type="DensityWeightedGridSample", 
             grid_size=final_grid_size,
             hash_type="fnv",
             mode="test",
             keys=("coord", "color"),
             return_grid_coord=True,
             max_grid_coord=num_bins,
             sampling_weights=sampling_weights),  # Custom transform
    ])
    
    # Fallback to regular GridSample if custom transform not available
    try:
        point_cloud = transform({
            "coord": points.copy(),
            "color": colors.copy(),
        })
    except:
        print("  Fallback to regular GridSample with adaptive grid size")
        fallback_transform = Compose([
            dict(type="PositiveShift"),
            dict(type="NormalizeColor"),
            dict(type="GridSample",
                 grid_size=final_grid_size,
                 hash_type="fnv", 
                 mode="test",
                 keys=("coord", "color"),
                 return_grid_coord=True,
                 max_grid_coord=num_bins),
        ])
        point_cloud = fallback_transform({
            "coord": points.copy(),
            "color": colors.copy(),
        })
    
    print(f"  Processed points: {len(point_cloud['grid_coord']):,}")
    
    # Convert to tensor format
    coord = point_cloud["grid_coord"]
    xyz = point_cloud["coord"]
    rgb = point_cloud["color"]
    point_cloud_data = np.concatenate([coord, xyz, rgb], axis=1)
    return torch.as_tensor(np.stack([point_cloud_data], axis=0))


def preprocess_point_cloud_adaptive(points, colors, base_grid_size, num_bins, density_levels=3):
    """
    Adaptive point cloud preprocessing with variable voxel density based on local point density.
    
    Args:
        points: Original point coordinates
        colors: Original point colors  
        base_grid_size: Base voxel size for sparse regions
        num_bins: Number of bins for coordinate discretization
        density_levels: Number of different density levels to use
    """
    from sklearn.neighbors import NearestNeighbors
    
    print(f"Adaptive preprocessing with {density_levels} density levels...")
    
    # Calculate local point density for each point
    k_neighbors = min(20, len(points) // 10)  # Adaptive k based on point cloud size
    nbrs = NearestNeighbors(n_neighbors=k_neighbors).fit(points)
    distances, _ = nbrs.kneighbors(points)
    
    # Use average distance to k-th neighbor as density measure (smaller = denser)
    local_density = 1.0 / (distances[:, -1] + 1e-6)  # Inverse distance = density
    
    # Create density-based regions
    density_percentiles = np.linspace(0, 100, density_levels + 1)
    density_thresholds = np.percentile(local_density, density_percentiles)
    
    processed_clouds = []
    total_points = 0
    
    for level in range(density_levels):
        # Define density range for this level
        min_density = density_thresholds[level]
        max_density = density_thresholds[level + 1]
        
        # Select points in this density range
        if level == density_levels - 1:  # Last level includes maximum
            mask = (local_density >= min_density)
        else:
            mask = (local_density >= min_density) & (local_density < max_density)
        
        if not np.any(mask):
            continue
            
        level_points = points[mask]
        level_colors = colors[mask]
        
        # Calculate grid size for this density level
        # Higher density regions get smaller voxels (more detail)
        density_factor = (level + 1) / density_levels  # 0.33, 0.67, 1.0 for 3 levels
        level_grid_size = base_grid_size * (1.0 / (density_factor ** 0.5))  # Square root scaling
        
        print(f"  Level {level + 1}: {np.sum(mask):,} points, grid_size={level_grid_size:.6f}")
        
        # Process this density level
        transform = Compose([
            dict(type="PositiveShift"),
            dict(type="NormalizeColor"),
            dict(
                type="GridSample",
                grid_size=level_grid_size,
                hash_type="fnv",
                mode="test",
                keys=("coord", "color"),
                return_grid_coord=True,
                max_grid_coord=num_bins,  # Use same coordinate space for all levels
            ),
        ])
        
        level_cloud = transform({
            "coord": level_points.copy(),
            "color": level_colors.copy(),
        })
        
        processed_clouds.append(level_cloud)
        total_points += len(level_cloud["grid_coord"])
    
    # Combine all density levels
    if len(processed_clouds) == 1:
        final_cloud = processed_clouds[0]
    else:
        final_cloud = combine_density_levels(processed_clouds, num_bins)
    
    print(f"  Total processed points: {total_points:,}")
    
    # Convert to tensor format
    coord = final_cloud["grid_coord"]
    xyz = final_cloud["coord"]
    rgb = final_cloud["color"]
    point_cloud = np.concatenate([coord, xyz, rgb], axis=1)
    return torch.as_tensor(np.stack([point_cloud], axis=0))


def combine_density_levels(processed_clouds, num_bins):
    """Combine multiple density-level point clouds into one unified coordinate system."""
    all_coords = []
    all_xyz = []
    all_colors = []
    
    for i, cloud in enumerate(processed_clouds):
        all_coords.append(cloud["grid_coord"])
        all_xyz.append(cloud["coord"])
        all_colors.append(cloud["color"])
    
    combined_cloud = {
        "grid_coord": np.vstack(all_coords),
        "coord": np.vstack(all_xyz),
        "color": np.vstack(all_colors),
    }
    
    # Remove duplicate voxels (keep the one from the highest density level)
    # Since we process from low to high density, later entries have higher priority
    grid_coords = combined_cloud["grid_coord"]
    
    # Create unique voxel keys
    voxel_keys = grid_coords[:, 0] * (num_bins ** 2) + grid_coords[:, 1] * num_bins + grid_coords[:, 2]
    
    # Keep last occurrence (highest density level) for each voxel
    _, unique_indices = np.unique(voxel_keys[::-1], return_index=True)
    unique_indices = len(voxel_keys) - 1 - unique_indices  # Reverse back to original indices
    unique_indices = np.sort(unique_indices)  # Sort to maintain order
    
    # Keep only unique voxels (prioritizing higher density levels)
    combined_cloud["grid_coord"] = combined_cloud["grid_coord"][unique_indices]
    combined_cloud["coord"] = combined_cloud["coord"][unique_indices]
    combined_cloud["color"] = combined_cloud["color"][unique_indices]
    
    print(f"  Combined: {len(unique_indices):,} unique voxels after deduplication")
    
    return combined_cloud


def preprocess_point_cloud(points, colors, grid_size, num_bins):
    transform = Compose(
        [
            dict(type="PositiveShift"),
            dict(type="NormalizeColor"),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="test",
                keys=("coord", "color"),
                return_grid_coord=True,
                max_grid_coord=num_bins,
            ),
        ]
    )
    point_cloud = transform(
        {
            "name": "pcd",
            "coord": points.copy(),
            "color": colors.copy(),
        }
    )
    coord = point_cloud["grid_coord"]
    xyz = point_cloud["coord"]
    rgb = point_cloud["color"]
    point_cloud = np.concatenate([coord, xyz, rgb], axis=1)
    return torch.as_tensor(np.stack([point_cloud], axis=0))


def generate_layout(
    model,
    point_cloud,
    tokenizer,
    code_template_file,
    top_k=10,
    top_p=0.95,
    temperature=0.6,
    num_beams=1,
    seed=-1,
    max_new_tokens=4096,
    detect_type="all",
    categories=[],
):
    if seed >= 0:
        set_seed(seed)

    # load the code template
    with open(code_template_file, "r") as f:
        code_template = f.read()

    task_prompt = DETECT_TYPE_PROMPT[detect_type]
    if detect_type != "arch" and categories:
        task_prompt = task_prompt.replace("boxes", ", ".join(categories))
    print("Task prompt: ", task_prompt)

    prompt = f"<|point_start|><|point_pad|><|point_end|>{task_prompt} The reference code is as followed: {code_template}"

    # prepare the conversation data
    conversation = [{"role": "user", "content": prompt}]

    input_ids = tokenizer.apply_chat_template(
        conversation, add_generation_prompt=True, return_tensors="pt"
    )
    input_ids = input_ids.to(model.device)

    # Choose between streaming and non-streaming based on num_beams
    if num_beams == 1:
        # Use streaming for single beam (default behavior)
        print("Using streaming generation (num_beams=1)")
        
        streamer = TextIteratorStreamer(
            tokenizer, timeout=20.0, skip_prompt=True, skip_special_tokens=True
        )

        generate_kwargs = dict(
            {"input_ids": input_ids, "point_clouds": point_cloud},
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            use_cache=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
        )
        t = Thread(target=model.generate, kwargs=generate_kwargs)
        t.start()

        print("Generating layout...\n")
        generate_texts = []
        for text in streamer:
            generate_texts.append(text)
            print(text, end="", flush=True)
        print("\nDone!")

        layout_str = "".join(generate_texts)
        
    else:
        # Use non-streaming for multiple beams (num_beams > 1)
        print(f"Using non-streaming generation (num_beams={num_beams})")
        
        generate_kwargs = dict(
            input_ids=input_ids,
            point_clouds=point_cloud,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            use_cache=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
        )
        
        print("Generating layout...\n")
        outputs = model.generate(**generate_kwargs)
        print("Done!")
        
        generated_ids = outputs[:, input_ids.shape[-1]:]
        layout_str = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    layout = Layout(layout_str)
    layout.undiscretize_and_unnormalize(num_bins=num_bins)
    return layout


if __name__ == "__main__":
    parser = argparse.ArgumentParser("SpatialLM inference script")
    parser.add_argument(
        "-p",
        "--point_cloud",
        type=str,
        required=True,
        help="Path to the input point cloud file or a folder containing multiple point cloud files",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Path to the output layout txt file or a folder to save multiple layout txt files",
    )
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        default="manycore-research/SpatialLM-Llama-1B",
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "-d",
        "--detect_type",
        type=str,
        default="all",
        choices=["all", "arch", "object"],
        help="The type of indoor elements to detect. all: (wall, door, window, box), arch: (wall, door, window), object: (box)",
    )
    parser.add_argument(
        "-c",
        "--category",
        nargs="+",
        default=[],
        choices=[
            "sofa",
            "chair",
            "dining_chair",
            "bar_chair",
            "stool",
            "bed",
            "pillow",
            "wardrobe",
            "nightstand",
            "tv_cabinet",
            "wine_cabinet",
            "bathroom_cabinet",
            "shoe_cabinet",
            "entrance_cabinet",
            "decorative_cabinet",
            "washing_cabinet",
            "wall_cabinet",
            "sideboard",
            "cupboard",
            "coffee_table",
            "dining_table",
            "side_table",
            "dressing_table",
            "desk",
            "integrated_stove",
            "gas_stove",
            "range_hood",
            "micro-wave_oven",
            "sink",
            "stove",
            "refrigerator",
            "hand_sink",
            "shower",
            "shower_room",
            "toilet",
            "tub",
            "illumination",
            "chandelier",
            "floor-standing_lamp",
            "wall_decoration",
            "painting",
            "curtain",
            "carpet",
            "plants",
            "potted_bonsai",
            "tv",
            "computer",
            "air_conditioner",
            "washing_machine",
            "clothes_rack",
            "mirror",
            "bookcase",
            "cushion",
            "bar",
            "screen",
            "combination_sofa",
            "dining_table_combination",
            "leisure_table_and_chair_combination",
            "multifunctional_combination_bed",
        ],
        help="A list of categories of objects to detect. If not specified, all categories will be detected.",
    )
    parser.add_argument(
        "-t",
        "--code_template_file",
        type=str,
        default="code_template.txt",
        help="Path to the code template file",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="The number of highest probability vocabulary tokens to keep for top-k filtering",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="The smallest set of most probable tokens with probabilities that add up to top_p or higher are kept",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="The value used to module the next token probabilities",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help="The number of beams for beam search",
    )
    parser.add_argument(
        "--inference_dtype",
        type=str,
        default="bfloat16",
        help="The torch dtype to use for inference, bfloat16 or float32",
    )
    parser.add_argument(
        "--no_cleanup",
        default=False,
        action="store_true",
        help="Whether to not cleanup the point cloud",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="The seed to use during inference, negative value means no seed",
    )
    parser.add_argument(
        "--save_processed_pcd",
        action="store_true",
        help="Save the processed point cloud as PLY to see what the model sees",
    )
    parser.add_argument(
        "--save_grid_pcd", 
        action="store_true",
        help="Save the grid coordinate point cloud as PLY to see the voxel structure",
    )
    parser.add_argument(
        "--voxel_density_multiplier",
        type=float,
        default=1.0,
        help="Multiply voxel count by this factor (2.0 = double voxels, default: 2.0)",
    )
    parser.add_argument(
        "--adaptive_sampling",
        action="store_true",
        help="Use adaptive voxel density based on local point density",
    )
    parser.add_argument(
        "--unified_adaptive",
        action="store_true", 
        help="Use unified adaptive sampling (single coordinate system, recommended)",
    )
    parser.add_argument(
        "--density_levels",
        type=int,
        default=1,
        help="Number of density levels for adaptive sampling (default: 3)",
    )
    parser.add_argument("--scale", action="store_true", help="Optimize wall scales to fit grid.ply perimeter")
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove intermediate artifacts (processed/grid PLYs, scale PNG). Final TXT remains.",
    )
    args = parser.parse_args()

    # load the model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=getattr(torch, args.inference_dtype)
    )
    model.to("cuda")
    model.set_point_backbone_dtype(torch.float32)
    model.eval()

    # number of bins used for discretization
    num_bins = model.config.point_config["num_bins"]
    print(f"Number of bins: {num_bins}")

    # check if the input is a single point cloud file or a folder containing multiple point cloud files
    if os.path.isfile(args.point_cloud):
        point_cloud_files = [args.point_cloud]
    else:
        point_cloud_files = glob.glob(os.path.join(args.point_cloud, "*.ply"))

    for point_cloud_file in tqdm(point_cloud_files):
        # Track intermediates for optional cleanup
        intermediates = []
        processed_ply_path = None
        grid_ply_path = None
        diagram_path = None
        base_filename = os.path.splitext(os.path.basename(point_cloud_file))[0]
        # load the point cloud
        point_cloud = load_o3d_pcd(point_cloud_file)
        # Use the standard grid size for coordinate system consistency
        # The model was trained with this grid size and expects this coordinate scaling
        grid_size = Layout.get_grid_size(num_bins)
        
        # For higher voxel density, we'll modify the cleanup_pcd voxel_size instead
        # This maintains coordinate system consistency while increasing point density
        cleanup_voxel_size = grid_size / args.voxel_density_multiplier
        print(f"Grid size: {grid_size:.6f} (standard), Cleanup voxel size: {cleanup_voxel_size:.6f} (density multiplier: {args.voxel_density_multiplier})")

        if not args.no_cleanup:
            point_cloud = cleanup_pcd(point_cloud, voxel_size=cleanup_voxel_size)

        points, colors = get_points_and_colors(point_cloud)
        min_extent = np.min(points, axis=0)

        # preprocess the point cloud to tensor features
        if args.unified_adaptive:
            input_pcd = preprocess_point_cloud_adaptive_unified(points, colors, grid_size, num_bins, args.density_levels)
        elif args.adaptive_sampling:
            input_pcd = preprocess_point_cloud_adaptive(points, colors, grid_size, num_bins, args.density_levels)
        else:
            input_pcd = preprocess_point_cloud(points, colors, grid_size, num_bins)

        # Save intermediate processed point cloud if requested
        if args.save_processed_pcd or args.save_grid_pcd:
            if args.save_processed_pcd:
                processed_ply_path = f"{base_filename}_processed.ply"
                save_processed_pointcloud(input_pcd, processed_ply_path, min_extent)
                intermediates.append(processed_ply_path)
                
            if args.save_grid_pcd:
                grid_ply_path = f"{base_filename}_grid.ply"
                save_grid_coordinates_pointcloud(input_pcd, grid_ply_path, grid_size_m=grid_size, min_extent=min_extent)
                intermediates.append(grid_ply_path)

        # generate the layout
        layout = generate_layout(
            model,
            input_pcd,
            tokenizer,
            args.code_template_file,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            num_beams=args.num_beams,
            seed=args.seed,
            detect_type=args.detect_type,
            categories=args.category,
        )
        layout.translate(min_extent)
        pred_language_string = layout.to_language_string()

        # Perform scale optimization if requested
        if args.scale:
            if not args.save_grid_pcd:
                print("Warning: --scale flag requires --save_grid_pcd to be active. Forcing grid PCD generation.")
                grid_ply_path = f"{base_filename}_grid.ply"
                save_grid_coordinates_pointcloud(input_pcd, grid_ply_path, grid_size_m=grid_size, min_extent=min_extent)
                intermediates.append(grid_ply_path)
            
            if os.path.exists(grid_ply_path):
                from scale_optimizer import run_scale_optimization
                output_dir = os.path.dirname(args.output) if os.path.splitext(args.output)[-1] else args.output
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                diagram_path = os.path.join(output_dir, f"{base_filename}_scale_optimization.png")
                intermediates.append(diagram_path)
                
                print(f"Running scale optimization with grid file: {grid_ply_path}")
                # Prefer using the original input PLY (meters) for perimeter extraction
                optimized_layout = run_scale_optimization(layout, grid_ply_path, diagram_path, source_ply_path=point_cloud_file)
                
                if optimized_layout:
                    print("Scale optimization successful. Using optimized layout.")
                    layout = optimized_layout
                    pred_language_string = layout.to_language_string()
                else:
                    print("Scale optimization failed. Using original layout.")
            else:
                print(f"Warning: Grid PLY file not found at {grid_ply_path}. Skipping scale optimization.")


        # check if the output path is a file or directory
        if os.path.splitext(args.output)[-1]:
            with open(args.output, "w") as f:
                f.write(pred_language_string)
        else:
            output_filename = os.path.basename(point_cloud_file).replace(".ply", ".txt")
            os.makedirs(args.output, exist_ok=True)
            with open(os.path.join(args.output, output_filename), "w") as f:
                f.write(pred_language_string)

        # Cleanup intermediates if requested
        if args.cleanup and intermediates:
            for p in set(intermediates):
                try:
                    if p and os.path.exists(p):
                        os.remove(p)
                        print(f"Cleaned up: {p}")
                except Exception as e:
                    print(f"Warning: Failed to remove {p}: {e}")
