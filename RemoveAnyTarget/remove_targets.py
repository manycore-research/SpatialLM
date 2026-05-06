#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

np = None
read_ply_elements = None
write_binary_ply = None


def load_runtime_deps():
    global np, read_ply_elements, write_binary_ply
    if np is not None:
        return
    import numpy as numpy
    from utils import read_ply_elements as read_elements
    from utils import write_binary_ply as write_ply

    np = numpy
    read_ply_elements = read_elements
    write_binary_ply = write_ply


DEFAULT_SOURCE_PLY = REPO_ROOT.parent / "data" / "InteriorGS" / "0001_839920" / "3dgs_standard.ply"
DEFAULT_SEG_ROOT = REPO_ROOT.parent / "data" / "segmented-GS"
DEFAULT_SCENE_ID = "0001_839920"
DEFAULT_SEG_SOURCE = "layout"


def resolve_path(path: str | None, default: Path | None = None) -> Path:
    raw_path = Path(path).expanduser() if path else default
    if raw_path is None:
        raise ValueError("path is required")
    if raw_path.is_absolute():
        return raw_path.resolve()
    return (Path.cwd() / raw_path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove selected segmented objects from a standard Gaussian PLY."
    )
    parser.add_argument(
        "--source-ply",
        type=str,
        default=None,
        help=f"Original standard Gaussian PLY. Default: {DEFAULT_SOURCE_PLY}",
    )
    parser.add_argument(
        "--seg-dir",
        type=str,
        default=None,
        help="Segmented scene directory containing object folders and manifest.json. Overrides --seg-source.",
    )
    parser.add_argument(
        "--seg-source",
        choices=("layout", "gt_labels"),
        default=DEFAULT_SEG_SOURCE,
        help="Which default segmentation result to use when --seg-dir is not set.",
    )
    parser.add_argument(
        "--scene-id",
        type=str,
        default=DEFAULT_SCENE_ID,
        help=f"Scene id used for default paths. Default: {DEFAULT_SCENE_ID}",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output PLY path. If omitted, writes to source scene removal_results/<seg_source>_<targets>/3dgs_removed.ply.",
    )
    parser.add_argument(
        "--remove",
        nargs="*",
        default=[],
        help="Object folder names to remove, e.g. 48_door 54_wardrobe.",
    )
    parser.add_argument(
        "--remove-file",
        type=str,
        default=None,
        help="Text file with one object folder name per line.",
    )
    parser.add_argument(
        "--remove-all",
        action="store_true",
        help="Remove all objects found in the selected segmentation directory.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available segmented object folders and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be removed without writing output PLY.",
    )
    return parser.parse_args()


def load_manifest(seg_dir: Path) -> dict:
    manifest_path = seg_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def list_objects(seg_dir: Path, manifest: dict):
    objects = manifest.get("objects") or []
    if objects:
        for obj in objects:
            folder = obj.get("folder", "")
            label = obj.get("label", "")
            count = obj.get("num_gaussians", "")
            print(f"{folder}\t{label}\t{count}")
        return

    for obj_dir in sorted(p for p in seg_dir.iterdir() if p.is_dir()):
        bbox_path = obj_dir / "bbox.json"
        label = ""
        count = ""
        if bbox_path.exists():
            bbox = json.loads(bbox_path.read_text(encoding="utf-8"))
            label = bbox.get("label", "")
            count = bbox.get("num_gaussians", "")
        print(f"{obj_dir.name}\t{label}\t{count}")


def all_object_names(seg_dir: Path, manifest: dict) -> list[str]:
    objects = manifest.get("objects") or []
    if objects:
        return [obj["folder"] for obj in objects if obj.get("folder")]
    return sorted(p.name for p in seg_dir.iterdir() if p.is_dir())


def load_remove_names(args: argparse.Namespace, seg_dir: Path, manifest: dict) -> list[str]:
    names = list(args.remove)
    if args.remove_all:
        names.extend(all_object_names(seg_dir, manifest))
    if args.remove_file:
        remove_file = resolve_path(args.remove_file)
        for line in remove_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                names.append(line)

    deduped = []
    seen = set()
    for name in names:
        if name not in seen:
            deduped.append(name)
            seen.add(name)
    return deduped


def load_indices(seg_dir: Path, object_names: list[str], vertex_count: int):
    load_runtime_deps()
    selected = []
    missing = []
    summaries = []

    for name in object_names:
        obj_dir = seg_dir / name
        idx_path = obj_dir / "indices.npy"
        if not idx_path.exists():
            missing.append(name)
            continue

        idx = np.load(idx_path).astype(np.int64, copy=False)
        if idx.ndim != 1:
            raise ValueError(f"indices.npy must be 1-D: {idx_path}")
        if idx.size and (idx.min() < 0 or idx.max() >= vertex_count):
            raise ValueError(
                f"indices out of bounds for {name}: min={idx.min()}, max={idx.max()}, vertices={vertex_count}"
            )

        selected.append(idx)
        summaries.append({"folder": name, "num_indices": int(idx.size)})

    if missing:
        available = sorted(p.name for p in seg_dir.iterdir() if p.is_dir())
        preview = ", ".join(available[:20])
        raise FileNotFoundError(
            f"Missing object folders or indices.npy: {', '.join(missing)}. "
            f"Available examples: {preview}"
        )

    if not selected:
        return np.array([], dtype=np.int64), summaries

    return np.unique(np.concatenate(selected)), summaries


def clean_name(value: str) -> str:
    value = re.sub(r"\s+", "_", value.strip().lower())
    value = re.sub(r"[^a-z0-9_-]+", "", value)
    return value or "target"


def default_output_path(
    source_ply: Path,
    seg_source: str,
    remove_names: list[str],
    remove_all: bool,
) -> Path:
    digest = hashlib.sha1("\n".join(remove_names).encode("utf-8")).hexdigest()[:8]
    if remove_all:
        slug = f"all_{digest}"
    elif len(remove_names) == 1:
        slug = clean_name(remove_names[0])
    else:
        slug = f"{len(remove_names)}targets_{digest}"
    result_name = f"{clean_name(seg_source)}_{slug}"
    return source_ply.parent / "removal_results" / result_name / "3dgs_removed.ply"


def default_seg_dir(seg_source: str, scene_id: str) -> Path:
    return DEFAULT_SEG_ROOT / seg_source / scene_id


def write_removed_ply(source_ply: Path, output_path: Path, remove_idx: np.ndarray):
    load_runtime_deps()
    elements, props_map = read_ply_elements(source_ply)
    vertices = elements["vertex"]
    vertex_count = len(vertices)

    keep_mask = np.ones(vertex_count, dtype=bool)
    keep_mask[remove_idx] = False

    out_elements = {}
    for name, arr in elements.items():
        if name == "vertex":
            out_elements[name] = arr[keep_mask]
        elif len(arr) == vertex_count:
            out_elements[name] = arr[keep_mask]
        else:
            out_elements[name] = arr

    ordered = [(name, out_elements[name], props_map[name]) for name in elements.keys()]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_binary_ply(output_path, ordered)
    return int(vertex_count), int(keep_mask.sum())


def main() -> int:
    args = parse_args()
    source_ply = resolve_path(args.source_ply, DEFAULT_SOURCE_PLY)
    seg_dir = resolve_path(args.seg_dir) if args.seg_dir else default_seg_dir(args.seg_source, args.scene_id)

    if not source_ply.exists():
        raise FileNotFoundError(f"source ply not found: {source_ply}")
    if not seg_dir.exists():
        raise FileNotFoundError(f"segmented dir not found: {seg_dir}")

    manifest = load_manifest(seg_dir)
    if args.list:
        list_objects(seg_dir, manifest)
        return 0

    load_runtime_deps()
    remove_names = load_remove_names(args, seg_dir, manifest)
    if not remove_names:
        raise ValueError("No objects specified. Use --remove, --remove-file, --remove-all, or --list.")
    output_path = (
        resolve_path(args.output)
        if args.output
        else default_output_path(source_ply, args.seg_source, remove_names, args.remove_all)
    )

    elements, _ = read_ply_elements(source_ply)
    vertex_count = len(elements["vertex"])
    remove_idx, summaries = load_indices(seg_dir, remove_names, vertex_count)

    report = {
        "source_ply": str(source_ply),
        "seg_dir": str(seg_dir),
        "seg_source": args.seg_source if not args.seg_dir else "custom",
        "output_ply": str(output_path),
        "num_vertices_source": int(vertex_count),
        "num_objects_requested": int(len(remove_names)),
        "num_vertices_to_remove": int(len(remove_idx)),
        "num_vertices_remaining": int(vertex_count - len(remove_idx)),
        "objects": summaries,
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0

    _, remaining = write_removed_ply(source_ply, output_path, remove_idx)
    report["num_vertices_written"] = remaining
    if args.output:
        manifest_path = output_path.with_suffix(".removed_manifest.json")
        targets_path = output_path.with_suffix(".removed_targets.txt")
    else:
        manifest_path = output_path.parent / "removed_manifest.json"
        targets_path = output_path.parent / "removed_targets.txt"
    manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    targets_path.write_text("\n".join(remove_names) + "\n", encoding="utf-8")
    print(f"[Done] wrote {output_path}")
    print(f"[Done] wrote {manifest_path}")
    print(f"[Done] wrote {targets_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
