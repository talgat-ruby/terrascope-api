"""Object detection on geo-referenced rasters.

Public surface:

- `Detection` / `Raster` / `Detector` — types and protocol (`types`).
- `build_detector` — pluggable detector factory (`factory`).
- `filter_detections` — confidence + AOI filter and id renumbering (`filter`).
- `render_overlay` — annotated PNG of bbox detections (`renderer`).
"""

from core.detection.factory import build_detector
from core.detection.filter import filter_detections
from core.detection.renderer import render_overlay
from core.detection.types import Detection, Detector, Raster

__all__ = [
    "Detection",
    "Detector",
    "Raster",
    "build_detector",
    "filter_detections",
    "render_overlay",
]
