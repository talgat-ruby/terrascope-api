"""Detection filter -- the entire postprocessor in one function.

Confidence threshold + (optional) AOI centroid containment + sequential id
renumbering (0..N). NMS, stitching, size filters, simplify -- intentionally
absent: SAHI handles overlap merging during inference, and the user does
not care about precision.
"""

from __future__ import annotations

from dataclasses import replace

from shapely.geometry.base import BaseGeometry

from core.detection.types import Detection


def filter_detections(
    detections: list[Detection],
    *,
    min_confidence: float = 0.25,
    aoi: BaseGeometry | None = None,
) -> list[Detection]:
    """Keep detections above `min_confidence` whose centroid lies inside `aoi`.

    `aoi` is optional; when omitted, no spatial filter applies. Surviving
    detections are renumbered sequentially starting at 0 in input order.
    """

    kept: list[Detection] = []
    for det in detections:
        if det.confidence < min_confidence:
            continue
        if aoi is not None and not (
            aoi.covers(det.centroid) or aoi.intersects(det.centroid)
        ):
            continue
        kept.append(det)

    return [replace(d, id=i) for i, d in enumerate(kept)]
