"""Detector factory.

Pluggable lookup keyed by detector `name`. Adding a new implementation is one
entry in `_BUILDERS`. The orchestrator never imports detector modules
directly.

Built-in detectors:

- `yolov8n-sahi`        COCO-pretrained YOLOv8n (80 generic classes).
                        Noisy on aerial imagery (TV/clock false positives).
- `yolov8-obb-aerial`   DOTA-pretrained YOLOv8s-OBB (15 aerial object
                        classes: plane, ship, vehicle, harbor, bridge, ...).
                        Does NOT include building / road / vegetation.
- `segformer-landscape` HuggingFace SegFormer-ADE20K. Per-pixel land-cover
                        segmentation -> bbox per connected component.
                        Surfaces building, road, grass, water, tree,
                        earth, sand, mountain.
- `aerial+landscape`    CompositeDetector running `yolov8-obb-aerial` and
                        `segformer-landscape` together. Yields both object
                        bboxes (ships, vehicles, planes) and land-cover
                        regions (buildings, roads, grass, water, ...).

To use a custom YOLO checkpoint, point `settings.yolo_weights` at any
Ultralytics-loadable `.pt` file.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from core.config import settings
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


def _build_segformer_landscape(**kwargs: Any) -> Detector:
    from core.detection.segformer_landscape import SegformerLandscapeDetector

    return SegformerLandscapeDetector(
        model_name=kwargs.get("model_name", settings.landscape_model),
        device=kwargs.get("device", settings.device),
        max_dim=kwargs.get("max_dim", settings.landscape_max_dim),
        min_pixels=kwargs.get("min_pixels", settings.landscape_min_pixels),
        name="segformer-landscape",
    )


def _build_aerial_plus_landscape(**kwargs: Any) -> Detector:
    from core.detection.composite import CompositeDetector

    return CompositeDetector(
        children=[
            _build_yolo_obb_aerial(**kwargs),
            _build_segformer_landscape(**kwargs),
        ],
        name="aerial+landscape",
    )


_BUILDERS: dict[str, Callable[..., Detector]] = {
    "yolov8n-sahi": _build_yolo_sahi,
    "yolov8-obb-aerial": _build_yolo_obb_aerial,
    "segformer-landscape": _build_segformer_landscape,
    "aerial+landscape": _build_aerial_plus_landscape,
}


def build_detector(name: str | None = None, **kwargs: Any) -> Detector:
    """Resolve `name` (or `settings.detector_name`) to a Detector instance."""
    key = name or settings.detector_name
    try:
        return _BUILDERS[key](**kwargs)
    except KeyError:
        known = ", ".join(sorted(_BUILDERS))
        raise ValueError(f"Unknown detector {key!r}. Known: {known}") from None
