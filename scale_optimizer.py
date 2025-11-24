import numpy as np
import open3d as o3d
from scipy.optimize import minimize, Bounds, LinearConstraint
from scipy.spatial import cKDTree

# Make matplotlib optional - if it's not available, diagrams won't be generated
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not found. Diagrams will not be generated.")

from copy import deepcopy

from spatiallm.layout.layout import Layout
from spatiallm.layout.entity import Wall

def huber_loss(distance: float, delta: float) -> float:
    """Huber loss for non-negative distance."""
    if distance <= delta:
        return 0.5 * distance * distance
    return delta * (distance - 0.5 * delta)

def point_to_segment_distance(point: np.ndarray, seg_start: np.ndarray, seg_end: np.ndarray) -> float:
    """Distance from a point to a 2D segment."""
    edge = seg_end - seg_start
    denom = float(np.dot(edge, edge))
    if denom == 0.0:
        return float(np.linalg.norm(point - seg_start))
    t = float(np.dot(point - seg_start, edge) / denom)
    t = max(0.0, min(1.0, t))
    closest = seg_start + t * edge
    return float(np.linalg.norm(point - closest))

def compute_principal_angle(points: np.ndarray) -> float:
    centered = points - np.mean(points, axis=0)
    cov = np.cov(centered.T)
    vals, vecs = np.linalg.eigh(cov)
    principal_vec = vecs[:, int(np.argmax(vals))]
    return float(np.arctan2(principal_vec[1], principal_vec[0]))

def rotate_vectors(vectors: np.ndarray, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    return (R @ vectors.T).T

def reconstruct_vertices_from_scales(scales: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    vertices = [np.zeros(2, dtype=float)]
    for i in range(len(scales)):
        vertices.append(vertices[-1] + scales[i] * vectors[i])
    return np.array(vertices[:-1])

def build_closure_matrix(vectors: np.ndarray) -> np.ndarray:
    if vectors.size == 0:
        return np.zeros((2, 0))
    return np.vstack([vectors[:, 0], vectors[:, 1]])

def compute_hull_edges(hull_points: np.ndarray):
    edges = []
    dirs = []
    angles = []
    H = len(hull_points)
    for i in range(H):
        p0 = hull_points[i]
        p1 = hull_points[(i + 1) % H]
        e = p1 - p0
        L = float(np.linalg.norm(e))
        if L == 0.0:
            u = np.array([1.0, 0.0])
        else:
            u = e / L
        edges.append((p0, p1))
        dirs.append(u)
        angles.append(float(np.arctan2(u[1], u[0])))
    return edges, np.array(dirs), np.array(angles)

def select_candidate_edges(vectors: np.ndarray, hull_dirs: np.ndarray, angle_threshold_deg: float = 20.0):
    candidates = []
    cos_thresh = float(np.cos(np.deg2rad(angle_threshold_deg)))
    for v in vectors:
        Lv = float(np.linalg.norm(v))
        if Lv == 0.0:
            uv = np.array([1.0, 0.0])
        else:
            uv = v / Lv
        dots = hull_dirs @ uv
        idx = np.where(np.abs(dots) >= cos_thresh)[0]
        if idx.size == 0:
            # fallback to best 3 by angle similarity
            idx = np.argsort(-np.abs(dots))[:3]
        candidates.append(idx.astype(int))
    return candidates

def make_objective(
    vectors: np.ndarray,
    hull_points: np.ndarray,
    num_samples_per_edge: int = 15,
    huber_delta: float = 0.04,
    reg_lambda: float = 3e-4,
    smooth_lambda: float = 1e-3,
    hull_edges: list | None = None,
    candidate_edges: list | None = None,
    use_kdtree: bool = True,
    k_neighbors: int = 8,
):
    # Precompute hull edges and candidates
    if hull_edges is None:
        hull_edges, hull_dirs, _ = compute_hull_edges(hull_points)
    else:
        # Recompute dirs for safety
        hull_dirs = []
        for (p0, p1) in hull_edges:
            e = p1 - p0
            L = float(np.linalg.norm(e))
            hull_dirs.append(e / L if L != 0.0 else np.array([1.0, 0.0]))
        hull_dirs = np.array(hull_dirs)
    if candidate_edges is None:
        candidate_edges = select_candidate_edges(vectors, hull_dirs)

    hull_center = np.mean(hull_points, axis=0)
    sample_ts = np.linspace(0.0, 1.0, num_samples_per_edge, dtype=float)

    # KD-tree over hull edge midpoints for spatial filtering
    if use_kdtree:
        mids = np.array([(e[0] + e[1]) * 0.5 for e in hull_edges])
        tree = cKDTree(mids)

    def objective(scales: np.ndarray) -> float:
        vertices = reconstruct_vertices_from_scales(scales, vectors)
        vertices_center = np.mean(vertices, axis=0)
        translation = hull_center - vertices_center
        vertices_aligned = vertices + translation

        total = 0.0
        # data term
        for i in range(len(scales)):
            start = vertices_aligned[i]
            end = start + scales[i] * vectors[i]
            cands = candidate_edges[i] if candidate_edges is not None else range(len(hull_edges))
            if use_kdtree:
                # spatial filter: nearest k edge midpoints to the segment midpoint
                seg_mid = (start + end) * 0.5
                _, idxs = tree.query(seg_mid, k=min(k_neighbors, len(hull_edges)))
                if np.isscalar(idxs):
                    idxs = np.array([int(idxs)])
                # intersect orientation and spatial candidates
                cands = np.intersect1d(np.array(cands, dtype=int), idxs.astype(int), assume_unique=False)
                if cands.size == 0:
                    cands = idxs.astype(int)
            for t in sample_ts:
                pt = (1.0 - t) * start + t * end
                min_d = float('inf')
                for j in cands:
                    h0, h1 = hull_edges[j]
                    d = point_to_segment_distance(pt, h0, h1)
                    if d < min_d:
                        min_d = d
                total += huber_loss(min_d, huber_delta)
        # L2 on deviation from 1
        diff = scales - 1.0
        total += reg_lambda * float(np.dot(diff, diff))
        # Smoothness across adjacent scales (circular)
        if smooth_lambda > 0.0 and len(scales) > 2:
            smooth = 0.0
            for i in range(len(scales)):
                prev_i = (i - 1) % len(scales)
                d = scales[i] - scales[prev_i]
                smooth += d * d
            total += smooth_lambda * float(smooth)
        return float(total)

    return objective

def transform_points_similarity(points: np.ndarray, theta: float, scale: float, translation: np.ndarray) -> np.ndarray:
    """Apply 2D similarity transform x' = s R x + t to a set of points.

    points: (N,2), translation: (2,)
    """
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return (scale * (R @ points.T)).T + translation

def transform_points_anisotropic(points: np.ndarray, theta: float, scale_x: float, scale_y: float, translation: np.ndarray) -> np.ndarray:
	"""Apply 2D anisotropic similarity transform x' = R diag(sx, sy) x + t."""
	c, s = np.cos(theta), np.sin(theta)
	R = np.array([[c, -s], [s, c]])
	D = np.array([[scale_x, 0.0], [0.0, scale_y]])
	return (R @ (D @ points.T)).T + translation

def build_segments_from_vertices(vertices: np.ndarray) -> np.ndarray:
    """Return segments as (M,2,2) array from polygon vertices."""
    if vertices.shape[0] < 2:
        return np.zeros((0, 2, 2))
    M = vertices.shape[0]
    segs = []
    for i in range(M):
        p0 = vertices[i]
        p1 = vertices[(i + 1) % M]
        segs.append([p0, p1])
    return np.array(segs)

def robust_align_hull_to_walls(hull_points: np.ndarray, orig_vertices: np.ndarray, huber_delta: float = 0.05):
    """Estimate similarity transform (theta, scale, tx, ty) aligning noisy hull to straight-line walls.

    Minimizes Huber-robust sum of distances from transformed hull points to the nearest original wall segment.
    """
    # Downsample hull for speed
    if len(hull_points) > 600:
        step = max(1, len(hull_points) // 600)
        hull_sample = hull_points[::step]
    else:
        hull_sample = hull_points

    segments = build_segments_from_vertices(orig_vertices)

    # Initial guess from PCA-based angle and RMS radius ratio
    hull_center = np.mean(hull_sample, axis=0)
    orig_center = np.mean(orig_vertices, axis=0)
    try:
        angle_h = compute_principal_angle(hull_sample)
        angle_o = compute_principal_angle(orig_vertices)
        theta0 = angle_o - angle_h
    except Exception:
        theta0 = 0.0
    r_o = float(np.sqrt(np.mean(np.sum((orig_vertices - orig_center) ** 2, axis=1))))
    r_h = float(np.sqrt(np.mean(np.sum((hull_sample - hull_center) ** 2, axis=1))))
    s0 = (r_o / r_h) if r_h > 1e-9 else 1.0
    # Translation to align centroids under initial rot-scale
    c0, s_ = np.cos(theta0), np.sin(theta0)
    R0 = np.array([[c0, -s_], [s_, c0]])
    t0 = orig_center - s0 * (R0 @ hull_center)

    def objective(p: np.ndarray) -> float:
        theta, scale, tx, ty = p
        transformed = transform_points_similarity(hull_sample, theta, max(scale, 1e-6), np.array([tx, ty]))
        total = 0.0
        for pt in transformed:
            # distance to nearest wall segment
            dmin = float('inf')
            for seg in segments:
                d = point_to_segment_distance(pt, seg[0], seg[1])
                if d < dmin:
                    dmin = d
            total += huber_loss(dmin, huber_delta)
        # small regularizer to keep scale near 1
        total += 1e-3 * (scale - 1.0) * (scale - 1.0)
        return float(total)

    # Bounds: theta free in [-pi, pi], scale in [0.5, 2.0], tx,ty wide bounds
    bounds = [(-np.pi, np.pi), (0.5, 2.0), (-1e6, 1e6), (-1e6, 1e6)]
    x0 = np.array([theta0, s0, t0[0], t0[1]], dtype=float)
    res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 300, 'ftol': 1e-12})
    theta_opt, scale_opt, tx_opt, ty_opt = res.x
    return float(theta_opt), float(scale_opt), np.array([tx_opt, ty_opt]), res

def robust_align_hull_to_walls_anisotropic(hull_points: np.ndarray, orig_vertices: np.ndarray, huber_delta: float = 0.05):
	"""Estimate anisotropic similarity (theta, sx, sy, tx, ty) aligning hull to straight walls.

	Minimizes Huber-robust sum of distances from transformed hull points to nearest wall segments.
	"""
	# Downsample hull for speed
	if len(hull_points) > 800:
		step = max(1, len(hull_points) // 800)
		hull_sample = hull_points[::step]
	else:
		hull_sample = hull_points

	segments = build_segments_from_vertices(orig_vertices)

	hull_center = np.mean(hull_sample, axis=0)
	orig_center = np.mean(orig_vertices, axis=0)
	try:
		angle_h = compute_principal_angle(hull_sample)
		angle_o = compute_principal_angle(orig_vertices)
		theta0 = angle_o - angle_h
	except Exception:
		theta0 = 0.0

	# Start from isotropic guess
	r_o = float(np.sqrt(np.mean(np.sum((orig_vertices - orig_center) ** 2, axis=1))))
	r_h = float(np.sqrt(np.mean(np.sum((hull_sample - hull_center) ** 2, axis=1))))
	s0 = (r_o / r_h) if r_h > 1e-9 else 1.0
	sx0, sy0 = s0, s0
	c0, s_ = np.cos(theta0), np.sin(theta0)
	R0 = np.array([[c0, -s_], [s_, c0]])
	t0 = orig_center - (R0 @ (np.array([[sx0, 0.0],[0.0, sy0]]) @ hull_center))

	def objective(p: np.ndarray) -> float:
		theta, log_sx, log_sy, tx, ty = p
		sx = float(np.exp(log_sx))
		sy = float(np.exp(log_sy))
		pts = transform_points_anisotropic(hull_sample, theta, sx, sy, np.array([tx, ty]))
		total = 0.0
		for pt in pts:
			# nearest segment distance
			dmin = float('inf')
			for seg in segments:
				d = point_to_segment_distance(pt, seg[0], seg[1])
				if d < dmin:
					dmin = d
			total += huber_loss(dmin, huber_delta)
		# mild regularization on anisotropy to avoid degenerate scaling
		total += 1e-3 * ((np.log(sx) - np.log(sy)) ** 2)
		return float(total)

	# Bounds and init
	bounds = [(-np.pi, np.pi), (np.log(0.5), np.log(2.0)), (np.log(0.5), np.log(2.0)), (-1e6, 1e6), (-1e6, 1e6)]
	x0 = np.array([theta0, np.log(sx0), np.log(sy0), t0[0], t0[1]], dtype=float)
	res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 400, 'ftol': 1e-12})
	theta_opt, log_sx_opt, log_sy_opt, tx_opt, ty_opt = res.x
	return float(theta_opt), float(np.exp(log_sx_opt)), float(np.exp(log_sy_opt)), np.array([tx_opt, ty_opt]), res

def order_walls_connected(walls, tol_primary: float = 0.05, tol_secondary: float = 0.15, allow_reverse: bool = True, inplace: bool = False):
    """
    Order walls so they form a connected chain with tolerant matching.

    - Prefer direct start match within tol_primary
    - Else allow reversing a wall and match within tol_primary
    - Else snap nearest candidate within tol_secondary by translating endpoints equally

    Returns (ordered_walls, is_closed).
    """
    if not walls:
        return [], False
    walls_in = list(walls) if inplace else [deepcopy(w) for w in walls]
    n = len(walls_in)
    if n == 1:
        return [walls_in[0]], True

    def d2(p1, p2):
        v = np.array(p1) - np.array(p2)
        return float(np.dot(v, v))

    ordered = [walls_in[0]]
    used = {0}

    for step in range(n - 1):
        last_wall = ordered[-1]
        last_end = np.array([last_wall.bx, last_wall.by])

        # 1) direct start match
        next_idx = -1
        min_dist2 = float('inf')
        for i, w in enumerate(walls_in):
            if i in used:
                continue
            dist2 = d2([w.ax, w.ay], last_end)
            if dist2 < min_dist2:
                min_dist2 = dist2
                next_idx = i
        if next_idx != -1 and np.sqrt(min_dist2) <= tol_primary:
            ordered.append(walls_in[next_idx])
            used.add(next_idx)
            continue

        # 2) reverse match
        if allow_reverse:
            next_idx_r = -1
            min_dist2_r = float('inf')
            for i, w in enumerate(walls_in):
                if i in used:
                    continue
                dist2 = d2([w.bx, w.by], last_end)
                if dist2 < min_dist2_r:
                    min_dist2_r = dist2
                    next_idx_r = i
            if next_idx_r != -1 and np.sqrt(min_dist2_r) <= tol_primary:
                wcopy = walls_in[next_idx_r]
                wcopy.ax, wcopy.bx = wcopy.bx, wcopy.ax
                wcopy.ay, wcopy.by = wcopy.by, wcopy.ay
                ordered.append(wcopy)
                used.add(next_idx_r)
                print(f"Connected by reversing wall at index {next_idx_r} (step {step}).")
                continue

        # 3) snap nearest within tol_secondary
        candidate_idx = -1
        best_is_start = True
        best_dist2 = float('inf')
        for i, w in enumerate(walls_in):
            if i in used:
                continue
            d_start = d2([w.ax, w.ay], last_end)
            d_end = d2([w.bx, w.by], last_end)
            if d_start < best_dist2:
                best_dist2 = d_start
                candidate_idx = i
                best_is_start = True
            if allow_reverse and d_end < best_dist2:
                best_dist2 = d_end
                candidate_idx = i
                best_is_start = False
        if candidate_idx != -1 and np.sqrt(best_dist2) <= tol_secondary:
            wcopy = walls_in[candidate_idx]
            if best_is_start:
                delta = last_end - np.array([wcopy.ax, wcopy.ay])
                wcopy.ax += delta[0]; wcopy.ay += delta[1]
                wcopy.bx += delta[0]; wcopy.by += delta[1]
                ordered.append(wcopy)
                used.add(candidate_idx)
                print(f"Snapped wall at index {candidate_idx} by {np.linalg.norm(delta):.3f} m to connect (step {step}).")
            else:
                wcopy.ax, wcopy.bx = wcopy.bx, wcopy.ax
                wcopy.ay, wcopy.by = wcopy.by, wcopy.ay
                delta = last_end - np.array([wcopy.ax, wcopy.ay])
                wcopy.ax += delta[0]; wcopy.ay += delta[1]
                wcopy.bx += delta[0]; wcopy.by += delta[1]
                ordered.append(wcopy)
                used.add(candidate_idx)
                print(f"Reversed+snapped wall at index {candidate_idx} by {np.linalg.norm(delta):.3f} m to connect (step {step}).")
            continue

        print(f"Warning: Walls are not connected! Missing connection after step {step} (wall id={getattr(last_wall, 'id', step)}).")
        return list(walls_in), False

    # Closure check and snap if close
    first_start = np.array([ordered[0].ax, ordered[0].ay])
    last_end = np.array([ordered[-1].bx, ordered[-1].by])
    gap = float(np.linalg.norm(first_start - last_end))
    if gap <= tol_secondary:
        delta = first_start - last_end
        ordered[-1].bx += delta[0]; ordered[-1].by += delta[1]
        if gap > tol_primary:
            print(f"Closed loop by snapping final gap of {gap:.3f} m.")
        return ordered, True

    print(f"Warning: Walls don't form closed loop! Gap: {gap:.6f}")
    return ordered, False

def extract_vectors_from_ordered_walls(ordered_walls):
    """
    Extract direction vectors from ordered walls.
    Ensures vectors follow the connectivity chain.
    """
    vectors = []
    vertices = []
    
    if not ordered_walls:
        return np.array([]), np.array([])
    
    # Start from first wall's start point
    start = np.array([ordered_walls[0].ax, ordered_walls[0].ay])
    vertices.append(start)
    
    for wall in ordered_walls:
        # Vector is always from start to end of this wall
        vec = np.array([wall.bx - wall.ax, wall.by - wall.ay])
        vectors.append(vec)
        # Next vertex is this wall's endpoint
        vertices.append(np.array([wall.bx, wall.by]))
    
    # Remove last vertex (it should equal the first for closed loop)
    vertices = vertices[:-1]
    
    return np.array(vertices), np.array(vectors)

def solve_for_closure(partial_scales, vectors, opt_indices, solve_indices):
    """
    Given N-2 scales, solve for the 2 remaining scales to ensure closed loop.
    """
    n = len(vectors)
    full_scales = np.ones(n)
    full_scales[opt_indices] = partial_scales
    
    # The constraint is that sum of all scaled vectors = 0
    # We have: sum(s_i * v_i for i in opt_indices) + s_a * v_a + s_b * v_b = 0
    # So: s_a * v_a + s_b * v_b = -sum(s_i * v_i for i in opt_indices)
    
    sum_opt = np.zeros(2)
    for i, idx in enumerate(opt_indices):
        sum_opt += partial_scales[i] * vectors[idx]
    
    # Build 2x2 system
    A = np.column_stack([vectors[solve_indices[0]], vectors[solve_indices[1]]])
    b = -sum_opt
    
    try:
        scales_solve = np.linalg.solve(A, b)
        full_scales[solve_indices[0]] = scales_solve[0]
        full_scales[solve_indices[1]] = scales_solve[1]
    except np.linalg.LinAlgError:
        print("Warning: Cannot solve for closure (walls may be parallel)")
        # Keep default scales of 1
    
    return full_scales

def objective_function(partial_scales, vectors, hull_points, opt_indices, solve_indices):
    """
    Minimize distance from wall vertices to hull perimeter.
    """
    # Get full scales with closure constraint
    scales = solve_for_closure(partial_scales, vectors, opt_indices, solve_indices)
    
    # Check if solved scales are valid (positive)
    if np.any(scales <= 0):
        return 1e10  # Large penalty for invalid scales
    
    # Build vertices from scaled vectors
    vertices = [np.zeros(2)]  # Start at origin (we'll translate later)
    for i, vec in enumerate(vectors):
        next_vertex = vertices[-1] + scales[i] * vec
        vertices.append(next_vertex)
    vertices = np.array(vertices[:-1])  # Remove duplicate last vertex
    
    # Center vertices to match hull
    vertices_center = np.mean(vertices, axis=0)
    hull_center = np.mean(hull_points, axis=0)
    vertices = vertices - vertices_center + hull_center
    
    # Calculate distance from vertices to hull
    total_dist = 0
    for v in vertices:
        min_dist = float('inf')
        for i in range(len(hull_points)):
            p1 = hull_points[i]
            p2 = hull_points[(i + 1) % len(hull_points)]
            # Distance from point to line segment
            edge = p2 - p1
            t = max(0, min(1, np.dot(v - p1, edge) / (np.dot(edge, edge) + 1e-10)))
            closest = p1 + t * edge
            dist = np.linalg.norm(v - closest)
            min_dist = min(min_dist, dist)
        total_dist += min_dist ** 2
    
    return total_dist

def generate_diagram(original_layout, optimized_layout, hull_points, output_path, meta_info: dict = None, hull_points_unscaled: np.ndarray | None = None):
    """Generate comparison diagram with overlays and vertex markers.

    meta_info (optional) may contain: 'scale_min', 'scale_mean', 'scale_max',
    'closure_residual', 'objective'.
    """
    if not HAS_MATPLOTLIB:
        print(f"Skipping diagram generation (matplotlib not available). Would have saved to: {output_path}")
        return
    
    plt.figure(figsize=(12, 10))
    
    # Plot hull
    hull_plot = np.vstack([hull_points, hull_points[0]])
    plt.plot(hull_plot[:, 0], hull_plot[:, 1], 'g.-', linewidth=2, markersize=3, label='Target Perimeter')

    # Plot unscaled/original hull in orange if provided
    if hull_points_unscaled is not None:
        hull_unscaled_plot = np.vstack([hull_points_unscaled, hull_points_unscaled[0]])
        plt.plot(hull_unscaled_plot[:, 0], hull_unscaled_plot[:, 1], color='orange', linestyle='-', linewidth=1.8, alpha=0.9, label='Target Perimeter (unscaled)')
    
    # Plot original walls
    for wall in original_layout.walls:
        plt.plot([wall.ax, wall.bx], [wall.ay, wall.by], 'b-', linewidth=2, alpha=0.7)
    plt.plot([], [], 'b-', linewidth=2, label='Original Walls')
    
    # Plot optimized/scale-applied walls
    if optimized_layout:
        for wall in optimized_layout.walls:
            plt.plot([wall.ax, wall.bx], [wall.ay, wall.by], 'r-', linewidth=2, alpha=0.85)
        plt.plot([], [], 'r-', linewidth=2, label='Scale-applied Walls')

    # Vertex markers for visual diff
    try:
        ow_ordered, _ = order_walls_connected(original_layout.walls)
        ov = []
        if ow_ordered:
            ov.append([ow_ordered[0].ax, ow_ordered[0].ay])
            for w in ow_ordered:
                ov.append([w.bx, w.by])
        ov = np.array(ov[:-1]) if len(ov) > 1 else np.array(ov)
        if ov.size:
            plt.scatter(ov[:, 0], ov[:, 1], c='b', s=10, alpha=0.6, marker='o', label='Original Vertices')
    except Exception:
        pass
    try:
        if optimized_layout:
            nw_ordered, _ = order_walls_connected(optimized_layout.walls)
            nv = []
            if nw_ordered:
                nv.append([nw_ordered[0].ax, nw_ordered[0].ay])
                for w in nw_ordered:
                    nv.append([w.bx, w.by])
            nv = np.array(nv[:-1]) if len(nv) > 1 else np.array(nv)
            if nv.size:
                plt.scatter(nv[:, 0], nv[:, 1], c='r', s=12, alpha=0.9, marker='x', label='Scaled Vertices')
    except Exception:
        pass
    
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    title = 'Wall Scale Optimization'
    if meta_info:
        parts = []
        if meta_info.get('scale_min') is not None and meta_info.get('scale_mean') is not None and meta_info.get('scale_max') is not None:
            parts.append(f"scales[min/mean/max]={meta_info['scale_min']:.3f}/{meta_info['scale_mean']:.3f}/{meta_info['scale_max']:.3f}")
        if meta_info.get('closure_residual') is not None:
            parts.append(f"||A s||={meta_info['closure_residual']:.2e}")
        if meta_info.get('objective') is not None:
            parts.append(f"obj={meta_info['objective']:.3f}")
        if parts:
            title += ' (' + ', '.join(parts) + ')'
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved diagram to: {output_path}")

def run_scale_optimization(layout: Layout, grid_ply_path: str, diagram_path: str, source_ply_path: str | None = None):
    """Main optimization function.

    If source_ply_path is provided, the perimeter/hull will be extracted from the original input PLY (meters).
    Otherwise, it will fall back to the grid PLY path.
    """
    # 1. Load and process grid PLY
    # Prefer original PLY (meters) if available
    ply_to_read = source_ply_path if source_ply_path else grid_ply_path
    try:
        pcd = o3d.io.read_point_cloud(ply_to_read)
        points = np.asarray(pcd.points)
        print(f"Loaded PLY for perimeter extraction: {ply_to_read}  (points={points.shape[0]})")
    except Exception as e:
        print(f"Error loading PLY '{ply_to_read}': {e}")
        return None

    if points.shape[0] < 3:
        print("Not enough points in grid PLY")
        return None

    # Filter by height to get wall slice
    z_coords = points[:, 2]
    z_min = np.min(z_coords)
    mask = (z_coords > z_min + 0.1) & (z_coords < z_min + 2.5)
    points_2d = points[mask, :2]

    if points_2d.shape[0] < 3:
        points_2d = points[:, :2]

    try:
        # Efficient perimeter sampling using fixed bins across bounding box
        n_bins = 100
        x = points_2d[:, 0]
        y = points_2d[:, 1]
        x_min, x_max = float(np.min(x)), float(np.max(x))
        y_min, y_max = float(np.min(y)), float(np.max(y))

        # Y-sliced extremes (min/max X per Y bin)
        bins_y = np.linspace(y_min, y_max, n_bins + 1)
        idx_y = np.digitize(y, bins_y) - 1
        idx_y = np.clip(idx_y, 0, n_bins - 1)
        min_x_by_bin = np.full(n_bins, np.inf)
        max_x_by_bin = np.full(n_bins, -np.inf)
        for b in range(n_bins):
            maskb = (idx_y == b)
            if not np.any(maskb):
                continue
            xb = x[maskb]
            min_x_by_bin[b] = float(np.min(xb))
            max_x_by_bin[b] = float(np.max(xb))
        y_centers = 0.5 * (bins_y[:-1] + bins_y[1:])

        perimeter_pts = []
        for b in range(n_bins):
            if np.isfinite(min_x_by_bin[b]):
                perimeter_pts.append([min_x_by_bin[b], y_centers[b]])
            if np.isfinite(max_x_by_bin[b]):
                perimeter_pts.append([max_x_by_bin[b], y_centers[b]])

        # X-sliced extremes (min/max Y per X bin)
        bins_x = np.linspace(x_min, x_max, n_bins + 1)
        idx_x = np.digitize(x, bins_x) - 1
        idx_x = np.clip(idx_x, 0, n_bins - 1)
        min_y_by_bin = np.full(n_bins, np.inf)
        max_y_by_bin = np.full(n_bins, -np.inf)
        for b in range(n_bins):
            maskb = (idx_x == b)
            if not np.any(maskb):
                continue
            yb = y[maskb]
            min_y_by_bin[b] = float(np.min(yb))
            max_y_by_bin[b] = float(np.max(yb))
        x_centers = 0.5 * (bins_x[:-1] + bins_x[1:])
        for b in range(n_bins):
            if np.isfinite(min_y_by_bin[b]):
                perimeter_pts.append([x_centers[b], min_y_by_bin[b]])
            if np.isfinite(max_y_by_bin[b]):
                perimeter_pts.append([x_centers[b], max_y_by_bin[b]])

        perimeter_pts = np.unique(np.array(perimeter_pts), axis=0)
        centroid = np.mean(perimeter_pts, axis=0)
        angles = np.arctan2(perimeter_pts[:, 1] - centroid[1], perimeter_pts[:, 0] - centroid[0])
        hull_points = perimeter_pts[np.argsort(angles)]
    except Exception as e:
        print(f"Error computing perimeter: {e}")
        return None

    # 2. Order walls by connectivity
    ordered_walls, connected = order_walls_connected(layout.walls)
    if not connected:
        print("ERROR: Input walls are not properly connected. Cannot optimize.")
        generate_diagram(layout, None, hull_points, diagram_path)
        return None

    # 3. Extract vertices and vectors from ordered walls
    orig_vertices, vectors = extract_vectors_from_ordered_walls(ordered_walls)
    if len(vectors) < 3:
        print("Need at least 3 walls to optimize")
        return None
    
    n = len(vectors)
    base_lengths = np.linalg.norm(vectors, axis=1)
    print(f"Connected loop with {n} walls. Base length stats (m): min={base_lengths.min():.3f} mean={base_lengths.mean():.3f} max={base_lengths.max():.3f}")

    # 4. Robust anisotropic pre-alignment: rotate + (sx, sy) scale + translate
    angle_align, sx_align, sy_align, t_align, res_align = robust_align_hull_to_walls_anisotropic(hull_points, orig_vertices, huber_delta=0.05)
    # Preserve unscaled (no scale) but rotated+translated hull for visualization
    hull_points_unscaled = transform_points_anisotropic(hull_points, angle_align, 1.0, 1.0, t_align)
    hull_points = transform_points_anisotropic(hull_points, angle_align, sx_align, sy_align, t_align)
    print(f"Robust anisotropic pre-align hull: rot={np.degrees(angle_align):.2f} deg, sx={sx_align:.4f}, sy={sy_align:.4f}, t=({t_align[0]:.3f},{t_align[1]:.3f}).")

    # 5. Keep wall directions in the original frame for optimization
    rotated_vectors = vectors
    print("Using original wall directions for optimization (no rotation/scale on vectors).")

    # 6. Build linear closure constraints
    A = build_closure_matrix(rotated_vectors)
    print(f"Using trivial per-axis scale from alignment; skipping nonlinear optimization.")

    # 7. Trivial per-axis: derive per-wall inverse scale to undo anisotropic factors
    # For wall i with unit direction u, inverse scale = sqrt((u_x/sx)^2 + (u_y/sy)^2)
    final_scales = np.zeros(n, dtype=float)
    for i in range(n):
        v = rotated_vectors[i]
        L = float(np.linalg.norm(v))
        u = (v / L) if L != 0.0 else np.array([1.0, 0.0])
        final_scales[i] = float(np.sqrt((u[0] / (sx_align if sx_align != 0.0 else 1.0))**2 + (u[1] / (sy_align if sy_align != 0.0 else 1.0))**2))
    eq_residual = A @ final_scales
    print(f"Closure residual (A @ s): ||A s||={np.linalg.norm(eq_residual):.3e}, components=({eq_residual[0]:.3e}, {eq_residual[1]:.3e})")

    # 8. Apply scales to layout (preserve original orientation; anchor at original first vertex)
    optimized_layout = deepcopy(layout)
    optimized_walls, _ = order_walls_connected(optimized_layout.walls, inplace=True)
    # Re-translate start to best match the aligned hull centroid to reduce drift
    orig_start = np.array([optimized_walls[0].ax, optimized_walls[0].ay, optimized_walls[0].az])
    current_pos = orig_start
    print("Applying scales to walls without rotation (preserving original orientation):")
    for i, wall in enumerate(optimized_walls):
        base_vec = vectors[i]
        length = float(np.linalg.norm(base_vec))
        if length == 0.0:
            direction = np.array([1.0, 0.0])
            new_length = 0.0
        else:
            direction = base_vec / length
            new_length = length * final_scales[i]
        wall.ax, wall.ay, wall.az = current_pos
        next_pos_2d = current_pos[:2] + direction * new_length
        wall.bx, wall.by = next_pos_2d
        wall.bz = wall.az
        current_pos = np.array([wall.bx, wall.by, wall.bz])
        print(f"  Wall {i} (id={getattr(wall, 'id', i)}): base_len={length:.3f} m, scale_inv={final_scales[i]:.4f} (undo sx={sx_align:.4f}, sy={sy_align:.4f}), new_len={new_length:.3f} m, start=({wall.ax:.3f},{wall.ay:.3f}), end=({wall.bx:.3f},{wall.by:.3f})")

    first_start = np.array([optimized_walls[0].ax, optimized_walls[0].ay])
    last_end = np.array([optimized_walls[-1].bx, optimized_walls[-1].by])
    closure_error = np.linalg.norm(first_start - last_end)
    if closure_error > 1e-3:
        print(f"WARNING: Closure error = {closure_error:.6f} meters")
    else:
        print(f"Closure check passed: residual distance between first/last = {closure_error:.6e} m")

	# 9. Diagram with meta
    meta = {
        'scale_min': float(final_scales.min()),
        'scale_mean': float(final_scales.mean()),
        'scale_max': float(final_scales.max()),
        'closure_residual': float(np.linalg.norm(eq_residual)),
        'objective': None,
    }
    generate_diagram(layout, optimized_layout, hull_points, diagram_path, meta_info=meta, hull_points_unscaled=hull_points_unscaled)
    return optimized_layout