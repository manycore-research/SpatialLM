#!/usr/bin/env python3

from __future__ import annotations

import argparse
from functools import partial
import hashlib
import json
import re
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import open3d as o3d

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parents[0]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spatiallm import Layout  # noqa: E402
from spatiallm.pcd import get_points_and_colors, load_o3d_pcd  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive Rerun Web Viewer for removing SpatialLM layout targets from a point cloud."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument(
        "-p",
        "--point-cloud",
        required=True,
        help="Path to the input point cloud file used by SpatialLM.",
    )
    parser.add_argument(
        "-l",
        "--layout",
        required=True,
        help="Path to the SpatialLM layout txt file.",
    )
    parser.add_argument(
        "--box-margin",
        type=float,
        default=0.05,
        help="Extra margin, in scene units, added to each layout box before removing points.",
    )
    parser.add_argument("--radius", type=float, default=0.01)
    parser.add_argument("--object-radius", type=float, default=0.018)
    parser.add_argument("--max-points-per-object", type=int, default=8000)
    parser.add_argument(
        "--max-scene-points",
        type=int,
        default=300000,
        help="Maximum number of scene points shown in the viewer. Use 0 to show all.",
    )
    return parser.parse_args()


def clean_rel_path(path: str) -> Path:
    rel = Path(path.lstrip("/"))
    if any(part == ".." for part in rel.parts):
        raise ValueError("parent traversal is not allowed")
    return rel


def clean_name(value: str) -> str:
    value = re.sub(r"\s+", "_", value.strip().lower())
    value = re.sub(r"[^a-z0-9_-]+", "", value)
    return value or "target"


def recording_url(handler: SimpleHTTPRequestHandler, rrd_path: Path) -> str:
    rel = rrd_path.relative_to(APP_DIR)
    host = handler.headers.get(
        "Host", f"{handler.server.server_address[0]}:{handler.server.server_address[1]}"
    )
    return f"http://{host}/files/{rel.as_posix()}"


def load_scene(config: dict) -> tuple[o3d.geometry.PointCloud, np.ndarray, np.ndarray]:
    pcd = load_o3d_pcd(config["point_cloud"])
    points, colors = get_points_and_colors(pcd)
    return pcd, points.astype(np.float32, copy=False), colors.astype(np.uint8, copy=False)


def load_layout_boxes(layout_path: str) -> list[dict]:
    layout_content = Path(layout_path).read_text(encoding="utf-8")
    boxes = Layout(layout_content).to_boxes()
    objects = []
    for box in boxes:
        label = str(box["label"])
        group = str(box["class"])
        folder = f"{int(box['id'])}_{clean_name(label)}"
        objects.append(
            {
                "folder": folder,
                "id": int(box["id"]),
                "label": label,
                "class": group,
                "center": np.asarray(box["center"], dtype=np.float32).tolist(),
                "scale": np.asarray(box["scale"], dtype=np.float32).tolist(),
                "rotation": np.asarray(box["rotation"], dtype=np.float32).tolist(),
            }
        )
    return objects


def object_lookup(config: dict) -> dict[str, dict]:
    return {obj["folder"]: obj for obj in load_layout_boxes(config["layout"])}


def points_inside_box(points: np.ndarray, obj: dict, margin: float) -> np.ndarray:
    center = np.asarray(obj["center"], dtype=np.float32)
    scale = np.asarray(obj["scale"], dtype=np.float32)
    rotation = np.asarray(obj["rotation"], dtype=np.float32)
    local = (points - center) @ rotation.T
    half_size = np.maximum(scale * 0.5 + margin, 0.0)
    return np.all(np.abs(local) <= half_size, axis=1)


def object_indices(points: np.ndarray, obj: dict, margin: float) -> np.ndarray:
    return np.flatnonzero(points_inside_box(points, obj, margin)).astype(np.int64, copy=False)


def downsample_indices(count: int, max_count: int) -> np.ndarray | None:
    if max_count <= 0 or count <= max_count:
        return None
    step = max(1, count // max_count)
    return np.arange(0, count, step, dtype=np.int64)[:max_count]


def write_point_cloud(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64, copy=False))
    if colors is not None and len(colors) == len(points):
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64, copy=False) / 255.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(path), pcd, write_ascii=False):
        raise RuntimeError(f"failed to write point cloud: {path}")


def make_preview_rrd_for_scene(
    points: np.ndarray,
    colors: np.ndarray,
    objects: list[dict],
    recording_name: str,
    config: dict,
) -> Path:
    import rerun as rr

    scene_indices = downsample_indices(len(points), config["max_scene_points"])
    if scene_indices is not None:
        scene_points = points[scene_indices]
        scene_colors = colors[scene_indices]
    else:
        scene_points = points
        scene_colors = colors

    recordings_dir = APP_DIR / ".runtime" / "recordings"
    recordings_dir.mkdir(parents=True, exist_ok=True)
    rrd_path = recordings_dir / f"{recording_name}.rrd"

    rr.init("remove_spatiallm_layout_target", spawn=False)
    rr.save(str(rrd_path))
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log(
        "world/scene",
        rr.Points3D(
            positions=scene_points,
            colors=scene_colors,
            radii=config["radius"],
        ),
        static=True,
    )

    margin = float(config["box_margin"])
    for obj in objects:
        folder = obj["folder"]
        center = np.asarray(obj["center"], dtype=np.float32)
        scale = np.asarray(obj["scale"], dtype=np.float32)
        rotation = np.asarray(obj["rotation"], dtype=np.float32)
        label = f"{folder} · {obj['label']}"
        entity = f"world/objects/{folder}"

        rr.log(
            f"{entity}/bbox",
            rr.Boxes3D(
                centers=[center],
                half_sizes=[np.maximum(scale * 0.5 + margin, 0.0)],
                labels=[label],
            ),
            rr.InstancePoses3D(mat3x3=[rotation]),
            static=True,
        )

        idx = object_indices(points, obj, margin)
        idx_indices = downsample_indices(len(idx), config["max_points_per_object"])
        if idx_indices is not None:
            idx = idx[idx_indices]
        if len(idx):
            rr.log(
                f"{entity}/points",
                rr.Points3D(
                    positions=points[idx],
                    colors=colors[idx],
                    radii=config["object_radius"],
                ),
                static=True,
            )

    return rrd_path


def make_preview_rrd(config: dict) -> Path:
    _, points, colors = load_scene(config)
    objects = load_layout_boxes(config["layout"])
    return make_preview_rrd_for_scene(
        points=points,
        colors=colors,
        objects=objects,
        recording_name=Path(config["layout"]).stem,
        config=config,
    )


def output_path_for(config: dict, targets: list[str]) -> Path:
    point_cloud = Path(config["point_cloud"])
    digest = hashlib.sha1("\n".join(targets).encode("utf-8")).hexdigest()[:8]
    slug = targets[0] if len(targets) == 1 else f"{len(targets)}targets_{digest}"
    return point_cloud.parent / "removal_results" / clean_name(slug) / f"{point_cloud.stem}_removed.ply"


def run_remove(config: dict, targets: list[str]) -> tuple[dict, Path]:
    _, points, colors = load_scene(config)
    lookup = object_lookup(config)
    missing = [target for target in targets if target not in lookup]
    if missing:
        raise FileNotFoundError(f"unknown layout target(s): {', '.join(missing)}")

    remove_mask = np.zeros(len(points), dtype=bool)
    summaries = []
    for target in targets:
        idx = object_indices(points, lookup[target], float(config["box_margin"]))
        remove_mask[idx] = True
        summaries.append({"folder": target, "label": lookup[target]["label"], "num_points": int(len(idx))})

    keep_mask = ~remove_mask
    output_path = output_path_for(config, targets)
    write_point_cloud(output_path, points[keep_mask], colors[keep_mask])

    report = {
        "point_cloud": config["point_cloud"],
        "layout": config["layout"],
        "output_ply": str(output_path),
        "box_margin": float(config["box_margin"]),
        "num_points_source": int(len(points)),
        "num_objects_requested": int(len(targets)),
        "num_points_to_remove": int(remove_mask.sum()),
        "num_points_remaining": int(keep_mask.sum()),
        "objects": summaries,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    (output_path.parent / "removed_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_path.parent / "removed_targets.txt").write_text("\n".join(targets) + "\n", encoding="utf-8")

    remaining_objects = [obj for obj in lookup.values() if obj["folder"] not in set(targets)]
    digest = hashlib.sha1("\n".join(targets).encode("utf-8")).hexdigest()[:8]
    rrd_path = make_preview_rrd_for_scene(
        points=points[keep_mask],
        colors=colors[keep_mask],
        objects=remaining_objects,
        recording_name=f"{Path(config['layout']).stem}_removed_{digest}",
        config=config,
    )
    return report, rrd_path


class Handler(SimpleHTTPRequestHandler):
    server_version = "RemoveSpatialLMTarget/0.1"

    def send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.path = "/static/index.html"
            return SimpleHTTPRequestHandler.do_GET(self)

        if parsed.path == "/api/config":
            cfg = self.server.config
            self.send_json(
                {
                    "mode": "spatiallm-layout",
                    "point_cloud": cfg["point_cloud"],
                    "layout": cfg["layout"],
                    "box_margin": cfg["box_margin"],
                }
            )
            return

        if parsed.path == "/api/objects":
            _, points, _ = load_scene(self.server.config)
            objects = load_layout_boxes(self.server.config["layout"])
            for obj in objects:
                obj["num_points"] = int(len(object_indices(points, obj, self.server.config["box_margin"])))
            self.send_json({"objects": objects})
            return

        if parsed.path == "/api/recording":
            rrd_path = make_preview_rrd(self.server.config)
            self.send_json({"url": recording_url(self, rrd_path)})
            return

        if parsed.path.startswith("/files/"):
            try:
                rel = clean_rel_path(parsed.path.removeprefix("/files/"))
            except ValueError as exc:
                self.send_error(400, str(exc))
                return
            self.path = "/" + rel.as_posix()
            return SimpleHTTPRequestHandler.do_GET(self)

        return SimpleHTTPRequestHandler.do_GET(self)

    def do_POST(self):
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")

        if parsed.path == "/api/remove":
            targets = payload.get("targets") or []
            if not isinstance(targets, list) or not all(isinstance(t, str) for t in targets):
                self.send_json({"error": "targets must be a string list"}, status=400)
                return
            try:
                report, rrd_path = run_remove(self.server.config, targets)
            except Exception as exc:
                self.send_json({"error": str(exc)}, status=500)
                return
            report["recording_url"] = recording_url(self, rrd_path)
            report["output_dir"] = str(Path(report["output_ply"]).parent)
            self.send_json(report)
            return

        self.send_json({"error": "not found"}, status=404)


def main() -> int:
    args = parse_args()
    point_cloud = Path(args.point_cloud).expanduser().resolve()
    layout = Path(args.layout).expanduser().resolve()
    if not point_cloud.exists():
        raise FileNotFoundError(f"point cloud not found: {point_cloud}")
    if not layout.exists():
        raise FileNotFoundError(f"layout not found: {layout}")

    config = {
        "point_cloud": str(point_cloud),
        "layout": str(layout),
        "box_margin": float(args.box_margin),
        "radius": float(args.radius),
        "object_radius": float(args.object_radius),
        "max_points_per_object": int(args.max_points_per_object),
        "max_scene_points": int(args.max_scene_points),
    }

    class AppServer(ThreadingHTTPServer):
        pass

    handler = partial(Handler, directory=str(APP_DIR))
    server = AppServer((args.host, args.port), handler)
    server.config = config
    print(f"[RemoveSpatialLMTarget] http://{args.host}:{args.port}")
    print(f"[RemoveSpatialLMTarget] point_cloud={point_cloud}")
    print(f"[RemoveSpatialLMTarget] layout={layout}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[RemoveSpatialLMTarget] stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
