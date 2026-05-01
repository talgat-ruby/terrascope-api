#!/usr/bin/env bash
# Sample CLI invocations for every registered detector preset.
#
# Each block runs `terrascope process` against $INPUT and writes outputs
# to its own directory under $OUT_BASE. Allowlists are scoped to the
# 7 viable classes from docs/classe-model.md:
#
#   building, road, car, park (vegetation), sports_field, bridge, water
#
# Each block's `classes` filter lists the *underlying model's native
# labels* that map into those 7 classes. Anything outside is dropped.
#
# Prerequisites:
#   - First run downloads model weights (Ultralytics .pt files cached in
#     ~/.cache or repo dir; HuggingFace models cached in ~/.cache/huggingface).
#   - `yolov8x-obb.pt` is a few hundred MB.
#   - Run `chmod +x scripts/sample_detect.sh` once before invoking directly.
#
# Usage:
#   1. Tweak INPUT below (Astana_1/2/3.tif or sample.tiff).
#   2. Uncomment the block(s) you want to run.
#   3. ./scripts/sample_detect.sh   (or: bash scripts/sample_detect.sh)
#
# Outputs per block: detections.geojson, overlay.png, indicators/ (if AOI).

set -euo pipefail

INPUT="inputs/Astana_4.tif"
OUT_BASE="outputs/Astana4_1"

# ---------- a) segformer-landscape (landcover only) ----------
# uv run terrascope process \
#   --input "$INPUT" \
#   --output "$OUT_BASE/segformer-landscape" \
#   --detectors '[
#     {"name": "segformer-landscape",
#      "classes": ["building", "road", "tree", "grass", "water"]}
#   ]'

# ---------- a2) aerial-road-segmenter (nadir/aerial-trained roads) ----------
# Reuses the SegFormer pipeline with a caller-supplied HF checkpoint and
# label_map. The default below uses Thalirajesh/Aerial-Drone-Image-Segmentation,
# a SegFormer-B0 finetuned on the Semantic Drone Dataset (Graz) — 24 classes
# including `paved-area`, `dirt`, `gravel` from drone/near-nadir altitudes.
# More appropriate for aerial road extraction than ADE20K-trained
# segformer-landscape / beit-ade, which were trained on street-level imagery
# and frequently miss roads entirely on nadir satellite/drone rasters.
#
# Other checkpoints worth trying: any SegFormer finetuned on DeepGlobe Roads,
# SpaceNet Roads, LoveDA, or Massachusetts Roads. Adjust label_map to match
# that checkpoint's id2label vocabulary.
uv run terrascope process \
  --input "$INPUT" \
  --output "$OUT_BASE/aerial-road-segmenter" \
  --detectors '[
    {"name": "aerial-road-segmenter",
     "classes": ["road"],
     "kwargs": {"model_name": "Thalirajesh/Aerial-Drone-Image-Segmentation",
                "label_map": {"paved-area": "road",
                              "gravel": "road",
                              "dirt": "road"}}}
  ]'

# ---------- b) beit-ade (same labels, larger backbone) ----------
# uv run terrascope process \
#   --input "$INPUT" \
#   --output "$OUT_BASE/beit-ade" \
#   --detectors '[
#     {"name": "beit-ade",
#      "classes": ["building", "road", "tree", "grass", "water"]}
#   ]'

# ---------- c) yolov8-obb-aerial (DOTA v1: cars + sports + bridge) ----------
# uv run terrascope process \
#   --input "$INPUT" \
#   --output "$OUT_BASE/yolov8-obb-aerial" \
#   --detectors '[
#     {"name": "yolov8-obb-aerial",
#      "classes": ["small-vehicle", "large-vehicle",
#                  "tennis-court", "basketball-court", "soccer-ball-field",
#                  "baseball-diamond", "ground-track-field",
#                  "bridge"]}
#   ]'

# ---------- d) yolov8-obb-dota-v2 (DOTA v2 OBB, same allowlist as c) ----------
# uv run terrascope process \
#   --input "$INPUT" \
#   --output "$OUT_BASE/yolov8-obb-dota-v2" \
#   --detectors '[
#     {"name": "yolov8-obb-dota-v2",
#      "classes": ["small-vehicle", "large-vehicle",
#                  "tennis-court", "basketball-court", "soccer-ball-field",
#                  "baseball-diamond", "ground-track-field",
#                  "bridge"]}
#   ]'

# ---------- e) yolov8-satellite-vehicle (HF keremberke, cars only) ----------
# uv run terrascope process \
#   --input "$INPUT" \
#   --output "$OUT_BASE/yolov8-satellite-vehicle" \
#   --detectors '[
#     {"name": "yolov8-satellite-vehicle",
#      "classes": ["vehicle", "car"],
#      "min_confidence": 0.3}
#   ]'

# ---------- f) yolov8n-sahi (COCO baseline, cars only) ----------
# uv run terrascope process \
#   --input "$INPUT" \
#   --output "$OUT_BASE/yolov8n-sahi" \
#   --detectors '[
#     {"name": "yolov8n-sahi",
#      "classes": ["car", "truck", "bus"]}
#   ]'

# ---------- g) combined: minimal 7-class setup (RECOMMENDED) ----------
# Landcover via SegFormer + objects via DOTA v2 OBB. Covers all 7 classes
# (building, road, park[tree+grass], water, car, sports_field, bridge).
# uv run terrascope process \
#   --input "$INPUT" \
#   --output "$OUT_BASE/combined" \
#   --detectors '[
#     {"name": "segformer-landscape",
#      "classes": ["building", "road", "tree", "grass", "water"]},
#     {"name": "yolov8-obb-dota-v2",
#      "classes": ["small-vehicle", "large-vehicle",
#                  "tennis-court", "basketball-court", "soccer-ball-field",
#                  "baseball-diamond", "ground-track-field",
#                  "bridge"]}
#   ]'
