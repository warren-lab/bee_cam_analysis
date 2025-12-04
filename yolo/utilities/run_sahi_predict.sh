#!/bin/bash

SOURCE_DIR="<SOURCE DIR>"
NAME_TAG="$(basename "$SOURCE_DIR")"

if [ ! -d "$SOURCE_DIR" ]; then
  echo "Source dir $SOURCE_DIR not found"
  exit 1
fi

echo "Running on $SOURCE_DIR → $NAME_TAG"

CUDA_VISIBLE_DEVICES=$GPU_ID \
sahi predict \
  --model_path <MODEL PATH> \
  --model_type ultralytics \
  --source "$SOURCE_DIR" \
  --slice_height 640 \
  --slice_width 640 \
  --overlap_height_ratio 0.2 \
  --overlap_width_ratio 0.2 \
  --project results/ \
  --novisual \
  --no-standard-prediction \
  --export_pickle \
  --name "$NAME_TAG" \
  2> "error_logs/sahi.err" || true
