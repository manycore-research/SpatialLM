#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

python web_app.py \
  --point-cloud ../../pcd/scene0000_00.ply \
  --layout ../../scene0000_00.txt \
  --host 127.0.0.1 \
  --port 7861



