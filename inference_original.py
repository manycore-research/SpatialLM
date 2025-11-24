import os
import glob
import argparse

import torch
import numpy as np
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
    if model.config.model_type == "spatiallm_qwen":
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    else:
        conversation = [{"role": "user", "content": prompt}]

    input_ids = tokenizer.apply_chat_template(
        conversation, add_generation_prompt=True, return_tensors="pt"
    )
    input_ids = input_ids.to(model.device)

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
    layout = Layout(layout_str)
    layout.undiscretize_and_unnormalize(num_bins=model.config.point_config["num_bins"])
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
        default=-1,
        help="The seed to use during inference, negative value means no seed",
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

    # check if the input is a single point cloud file or a folder containing multiple point cloud files
    if os.path.isfile(args.point_cloud):
        point_cloud_files = [args.point_cloud]
    else:
        point_cloud_files = glob.glob(os.path.join(args.point_cloud, "*.ply"))

    for point_cloud_file in tqdm(point_cloud_files):
        # load the point cloud
        point_cloud = load_o3d_pcd(point_cloud_file)
        grid_size = Layout.get_grid_size(num_bins)

        if not args.no_cleanup:
            point_cloud = cleanup_pcd(point_cloud, voxel_size=grid_size)

        points, colors = get_points_and_colors(point_cloud)
        min_extent = np.min(points, axis=0)

        # preprocess the point cloud to tensor features
        input_pcd = preprocess_point_cloud(points, colors, grid_size, num_bins)

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

        # check if the output path is a file or directory
        if os.path.splitext(args.output)[-1]:
            with open(args.output, "w") as f:
                f.write(pred_language_string)
        else:
            output_filename = os.path.basename(point_cloud_file).replace(".ply", ".txt")
            os.makedirs(args.output, exist_ok=True)
            with open(os.path.join(args.output, output_filename), "w") as f:
                f.write(pred_language_string)
