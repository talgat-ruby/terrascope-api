"""Object detection on geo-referenced rasters.

Public surface:

- `Detection` / `Raster` / `Detector` — types and protocol (`types`).
- `DetectorSpec` — declarative per-detector job config (`spec`).
- `build_from_specs` — pluggable detector factory (`factory`).
- `filter_detections` — confidence + AOI filter and id renumbering (`filter`).
- `render_overlay` — annotated PNG of bbox detections (`renderer`).
"""

from core.detection.factory import build_from_specs
from core.detection.filter import filter_detections
from core.detection.renderer import render_overlay
from core.detection.spec import DetectorSpec
from core.detection.types import Detection, Detector, Raster

__all__ = [
    "Detection",
    "Detector",
    "DetectorSpec",
    "Raster",
    "build_from_specs",
    "filter_detections",
    "render_overlay",
]
