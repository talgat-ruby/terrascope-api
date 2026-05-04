"""Post-detection filtering helpers.

Cheap, generic filters that don't belong to any single Process — kept in
`core.detection` so callers can re-run them outside the orchestrator
(e.g. when loading a previously-emitted GeoJSON for replay).
"""

from __future__ import annotations

from collections.abc import Iterable

from shapely.geometry.base import BaseGeometry

from core.detection.types import Detection


def filter_detections(
    detections: Iterable[Detection],
    *,
    min_confidence: float | None = None,
    aoi: BaseGeometry | None = None,
) -> list[Detection]:
    """Drop detections under `min_confidence` or outside `aoi`.

    `aoi`, when given, is matched against the detection's WGS84 `bbox` via
    `shapely.intersects` — both `geometry` and `bbox` are accepted, but the
    bbox is enough for the cheap pre-filter callers want here.
    """
    out: list[Detection] = []
    for det in detections:
        if min_confidence is not None and det.confidence < min_confidence:
            continue
        if aoi is not None:
            from shapely.geometry import box

            if not box(*det.bbox).intersects(aoi):
                continue
        out.append(det)
    return out


__all__ = ["filter_detections"]
