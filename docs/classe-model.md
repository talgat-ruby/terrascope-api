# Class → Pretrained Model Candidates

Per-class shortlist of pretrained models (min. 2 each) for the viable classes
from `docs/classes_analysis.md`. Excludes `people` and `mountain` (deemed
unviable at the project's GSD).

## 1. Building
- `microsoft/beit-large-finetuned-ade-640-640` (HuggingFace) — **registered as `beit-ade`**
- `nvidia/segformer-b0-finetuned-ade-512-512` — **registered as `segformer-landscape`**
- `SegFormer-B5` finetuned on INRIA Aerial _(no HF preset wired)_
- `ResUNet++` on WHU Building _(no HF preset wired)_
- `SAM` + finetune via `samgeo` _(no HF preset wired)_

## 2. Road
- `aerial-road-segmenter` — **recommended for nadir imagery.** Reuses the
  HF SemanticSegmentation pipeline; pass any aerial-road-finetuned
  checkpoint via `kwargs.model_name` plus a `kwargs.label_map` mapping
  the checkpoint's `id2label` strings to `road`.
- `segformer-landscape` (ADE20K — surfaces `road`, `sidewalk`, `path`, `runway`).
  ADE20K is street-level/oblique; on true nadir often misses roads entirely.
- `beit-ade` (same label space, larger backbone) — same domain caveat.
- `D-LinkNet` (DeepGlobe Challenge 2018) _(custom backend required)_
- `SegFormer` finetuned on DeepGlobe _(custom finetune required)_

## 3. Car
- `yolov8-satellite-vehicle` (HF `keremberke/yolov8m-satellite-vehicle-detection`)
- `yolov8-obb-aerial` (DOTA v1, `small-vehicle` / `large-vehicle`)
- `yolov8-obb-dota-v2` (DOTA v2, vehicle subclasses)
- `RT-DETR` on xView _(custom backend required)_

## 4. Park / Vegetation
- `segformer-landscape` (ADE20K — `grass`, `tree`, `plant`, `palm`, `field`)
- `beit-ade` (same labels, higher capacity)
- `SegFormer` finetuned on ISPRS Potsdam _(custom finetune required)_

## 5. Sports Field / Court
- `yolov8-obb-dota-v2` (DOTA v2: `tennis-court`, `basketball-court`, `soccer-ball-field`, `baseball-diamond`, `ground-track-field`)
- `yolov8-obb-aerial` (DOTA v1, narrower set)

## 6. Bridge
- `yolov8-obb-aerial` (DOTA v1 `bridge`)
- `yolov8-obb-dota-v2` (DOTA v2 `bridge`)
- `Faster R-CNN` on DIOR _(custom backend required)_

## 7. Water
- `segformer-landscape` (ADE20K — `water`, `river`, `lake`, `sea`, `pool`, `fountain`)
- `beit-ade` (same labels, higher capacity)
- NDWI-RGB threshold as a non-ML baseline: `(G - R) / (G + R)` _(no preset)_

---

## Minimal practical setup (registered)

```json
[
  {"name": "segformer-landscape",
   "classes": ["building", "road", "tree", "grass", "water"]},
  {"name": "yolov8-obb-dota-v2",
   "classes": ["small-vehicle", "large-vehicle",
               "tennis-court", "basketball-court", "soccer-ball-field",
               "baseball-diamond", "ground-track-field", "bridge"]}
]
```

Swap `segformer-landscape` for `beit-ade` if you want a higher-capacity
segmenter at the cost of inference speed. Swap `yolov8-obb-dota-v2` →
`yolov8-satellite-vehicle` if you only need cars and want a tighter
checkpoint for that class.

## Registered factory keys

| Key                        | Backend / weights                                         |
|----------------------------|-----------------------------------------------------------|
| `yolov8n-sahi`             | `yolov8n.pt` (COCO)                                       |
| `yolov8-obb-aerial`        | `yolov8s-obb.pt` (DOTA v1)                                |
| `yolov8-obb-dota-v2`       | `yolov8x-obb.pt` (DOTA v2)                                |
| `yolov8-satellite-vehicle` | `keremberke/yolov8m-satellite-vehicle-detection`          |
| `segformer-landscape`      | `nvidia/segformer-b0-finetuned-ade-512-512`               |
| `beit-ade`                 | `microsoft/beit-large-finetuned-ade-640-640`              |
| `aerial-road-segmenter`    | user-supplied HF checkpoint (nadir-trained roads)         |

Anything marked _custom backend required_ above (D-LinkNet, RT-DETR,
Faster R-CNN-DIOR, ISPRS-Potsdam-finetuned SegFormer, etc.) needs a new
detector class — they don't fit the existing YOLO+SAHI or HF
SemanticSegmentation backends without code changes.
