# RemoveAnyTarget

Interactive Rerun Web Viewer for removing objects from a SpatialLM point cloud using
the predicted layout boxes.

This version is for **ordinary point clouds** such as `point_cloud_rgb.ply`.
It does not preserve 3D Gaussian attributes from files like `3dgs_standard.ply`.

## Quick Start

Run from this folder:

```bash
cd RemoveAnyTarget
./run_web_app.sh
```

Then open:

```text
http://127.0.0.1:7861
```

The default script uses:

```text
--point-cloud ../pcd/scene0000_00.ply
--layout      ../scene0000_00.txt
--host        127.0.0.1
--port        7861
```

Edit `run_web_app.sh` if your files are somewhere else, for example:

```bash
python web_app.py \
  --point-cloud ../../data/InteriorGS/0001_839920/point_cloud_rgb.ply \
  --layout ../scene0000_00.txt \
  --host 127.0.0.1 \
  --port 7861
```

`run_remove_targets.sh` is kept as a convenience wrapper and starts the same Web UI.

## How It Works

The app loads:

- a point cloud PLY
- a SpatialLM layout txt file

It parses the layout with `Layout(...).to_boxes()`, creates one selectable Rerun
entity per predicted box, and removes points that fall inside the selected boxes.

Rerun entities are logged under:

```text
world/objects/{object_id}/bbox
world/objects/{object_id}/points
```

In the browser, click an object box or object point group in the Rerun viewer. The
matching checkbox should be selected in the left panel. You can also tick objects
manually, then click **Export removed scene**.

## Output

Removed point clouds are written next to the input point cloud:

```text
<point_cloud_dir>/removal_results/<target_name>/<point_cloud_name>_removed.ply
```

Each export also writes:

```text
removed_manifest.json
removed_targets.txt
```

## Useful Options

```bash
python web_app.py \
  --point-cloud ../pcd/scene0000_00.ply \
  --layout ../scene0000_00.txt \
  --box-margin 0.05 \
  --max-scene-points 300000 \
  --max-points-per-object 8000
```

`--box-margin` expands each layout box before removing points. Increase it if too
many object points remain; decrease it if neighboring objects are being removed.

`--max-scene-points` and `--max-points-per-object` only affect the viewer preview,
not the exported removed point cloud.

## Notes

This app removes points by SpatialLM layout boxes, so the result depends on box
accuracy. It is fast and simple, but not as precise as instance mask based removal.

For 3D Gaussian PLY files, use a pipeline that preserves all Gaussian vertex
properties. This Web UI currently writes ordinary point cloud PLY files through
Open3D.
