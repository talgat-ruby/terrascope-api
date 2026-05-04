"""Detection types + the lightweight helpers shared across processes."""

from core.detection.filters import filter_detections
from core.detection.types import Detection, Raster

__all__ = ["Detection", "Raster", "filter_detections"]
