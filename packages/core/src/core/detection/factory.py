"""Detector factory.

Pluggable lookup keyed by detector `name`. Adding a new implementation is one
entry in `_BUILDERS`. Multi-model orchestration is expressed via a list of
`DetectorSpec` and assembled by `build_from_specs`.

Built-in leaf detectors:

- `yolov8n-sahi`        COCO-pretrained YOLOv8n (80 generic classes).
- `yolov8-obb-aerial`   DOTA-pretrained YOLOv8s-OBB (15 aerial classes:
                        plane, ship, vehicle, harbor, bridge, ...).
- `yolov8-satellite-vehicle`  HuggingFace
                        `keremberke/yolov8m-satellite-vehicle-detection`
                        finetuned for cars on satellite imagery.
- `yolov8-obb-dota-v2`  Ultralytics DOTA v2 OBB (sports fields, bridge,
                        small/large-vehicle, harbor, ...).
- `segformer-landscape` HuggingFace SegFormer-ADE20K landscape segmenter.
- `beit-ade`            HuggingFace `microsoft/beit-large-finetuned-ade-640-640`
                        — higher-capacity ADE20K segmenter (slower than
                        segformer-b0, but better building/vegetation recall).

Multi-class presets typically combine a segmenter (for area classes:
building, road, vegetation, water) with an OBB detector (for object classes:
car, sports field, bridge). See `docs/classe-model.md`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from core.config import settings
from core.detection.composite import CompositeDetector
from core.detection.spec import DetectorSpec
from core.detection.types import Detector


def _build_yolo_sahi(**kwargs: Any) -> Detector:
    from core.detection.yolo_sahi import YoloSahiDetector

    return YoloSahiDetector(
        weights=kwargs.get("weights", settings.yolo_weights),
        device=kwargs.get("device", settings.device),
        confidence_threshold=kwargs.get(
            "confidence_threshold", settings.min_confidence
        ),
        name="yolov8n-sahi",
    )


def _build_yolo_obb_aerial(**kwargs: Any) -> Detector:
    from core.detection.yolo_sahi import YoloSahiDetector

    return YoloSahiDetector(
        weights=kwargs.get("weights", "yolov8s-obb.pt"),
        device=kwargs.get("device", settings.device),
        confidence_threshold=kwargs.get(
            "confidence_threshold", settings.min_confidence
        ),
        name="yolov8-obb-aerial",
    )


def _build_yolo_obb_dota_v2(**kwargs: Any) -> Detector:
    """DOTA v2 OBB checkpoint (more sports/vehicle subclasses than v1)."""
    from core.detection.yolo_sahi import YoloSahiDetector

    return YoloSahiDetector(
        weights=kwargs.get("weights", "yolov8x-obb.pt"),
        device=kwargs.get("device", settings.device),
        confidence_threshold=kwargs.get(
            "confidence_threshold", settings.min_confidence
        ),
        name="yolov8-obb-dota-v2",
    )


def _build_yolo_satellite_vehicle(**kwargs: Any) -> Detector:
    """Keremberke's YOLOv8m finetuned on satellite vehicle imagery."""
    from core.detection.yolo_sahi import YoloSahiDetector

    return YoloSahiDetector(
        weights=kwargs.get(
            "weights", "keremberke/yolov8m-satellite-vehicle-detection"
        ),
        device=kwargs.get("device", settings.device),
        confidence_threshold=kwargs.get(
            "confidence_threshold", settings.min_confidence
        ),
        name="yolov8-satellite-vehicle",
    )


def _build_segformer_landscape(**kwargs: Any) -> Detector:
    from core.detection.segformer_landscape import SegformerLandscapeDetector

    return SegformerLandscapeDetector(
        model_name=kwargs.get("model_name", settings.landscape_model),
        device=kwargs.get("device", settings.device),
        max_dim=kwargs.get("max_dim", settings.landscape_max_dim),
        min_pixels=kwargs.get("min_pixels", settings.landscape_min_pixels),
        name="segformer-landscape",
    )


def _build_beit_ade(**kwargs: Any) -> Detector:
    """BEiT-large finetuned on ADE20K — higher-capacity drop-in replacement.

    Reuses the SegformerLandscapeDetector pipeline (HF
    AutoModelForSemanticSegmentation + AutoImageProcessor); the ADE20K label
    space is the same, so the existing `_ADE_LANDSCAPE_LABELS` mapping
    applies as-is.
    """
    from core.detection.segformer_landscape import SegformerLandscapeDetector

    return SegformerLandscapeDetector(
        model_name=kwargs.get(
            "model_name", "microsoft/beit-large-finetuned-ade-640-640"
        ),
        device=kwargs.get("device", settings.device),
        max_dim=kwargs.get("max_dim", settings.landscape_max_dim),
        min_pixels=kwargs.get("min_pixels", settings.landscape_min_pixels),
        name="beit-ade",
    )


def _build_aerial_road(**kwargs: Any) -> Detector:
    """Nadir-trained road segmenter.

    Reuses the SegformerLandscapeDetector pipeline with a caller-supplied
    `model_name` (a HF checkpoint finetuned on aerial/satellite road data)
    and a caller-supplied `label_map` mapping the checkpoint's id2label
    strings onto the unified `road` class.
    """
    from core.detection.segformer_landscape import SegformerLandscapeDetector

    model_name = kwargs.get("model_name") or settings.aerial_road_model
    if not model_name:
        raise ValueError(
            "aerial-road-segmenter requires kwargs.model_name "
            "(or settings.aerial_road_model) — pick a HF checkpoint "
            "trained on nadir/aerial road data"
        )
    label_map = kwargs.get("label_map") or {"road": "road"}
    return SegformerLandscapeDetector(
        model_name=model_name,
        device=kwargs.get("device", settings.device),
        max_dim=kwargs.get("max_dim", settings.aerial_road_max_dim),
        min_pixels=kwargs.get("min_pixels", settings.aerial_road_min_pixels),
        name="aerial-road-segmenter",
        label_map=label_map,
    )


_BUILDERS: dict[str, Callable[..., Detector]] = {
    "yolov8n-sahi": _build_yolo_sahi,
    "yolov8-obb-aerial": _build_yolo_obb_aerial,
    "yolov8-obb-dota-v2": _build_yolo_obb_dota_v2,
    "yolov8-satellite-vehicle": _build_yolo_satellite_vehicle,
    "segformer-landscape": _build_segformer_landscape,
    "beit-ade": _build_beit_ade,
    "aerial-road-segmenter": _build_aerial_road,
}


def build_leaf(spec: DetectorSpec) -> Detector:
    """Resolve a single spec to a leaf Detector instance."""
    try:
        builder = _BUILDERS[spec.name]
    except KeyError:
        known = ", ".join(sorted(_BUILDERS))
        raise ValueError(
            f"Unknown detector {spec.name!r}. Known: {known}"
        ) from None
    return builder(**spec.kwargs)


def build_from_specs(specs: list[DetectorSpec]) -> Detector:
    """Build a Detector from a list of specs.

    Always wraps in a `CompositeDetector` so per-spec class allowlists,
    confidence overrides, and `source_model` provenance stamping are
    enforced uniformly — single-spec jobs go through the same path as
    multi-spec jobs.
    """
    if not specs:
        raise ValueError("build_from_specs requires at least one spec")
    pairs = [(build_leaf(spec), spec) for spec in specs]
    return CompositeDetector(pairs=pairs)
