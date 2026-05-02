"""Detection filter -- the entire postprocessor in one function.

Confidence threshold + (optional) AOI centroid containment + sequential id
renumbering (0..N). NMS, stitching, size filters, simplify -- intentionally
absent: SAHI handles overlap merging during inference, and the user does
not care about precision.
"""

from __future__ import annotations

from dataclasses import replace

from shapely.geometry.base import BaseGeometry

from core.detection.spec import DetectorSpec
from core.detection.types import Detection


def filter_detections(
    detections: list[Detection],
    *,
    min_confidence: float = 0.25,
    aoi: BaseGeometry | None = None,
    specs: list[DetectorSpec] | None = None,
) -> list[Detection]:
    """Keep detections above `min_confidence` whose centroid lies inside `aoi`.

    `min_confidence` is the *default* floor: it applies only to detections
    whose source spec did not set its own `min_confidence`. When `specs` is
    supplied and a detection's `source_model` matches a spec with an
    explicit `min_confidence`, that per-spec value (already applied inside
    `CompositeDetector`) wins and the global default is skipped — so users
    can intentionally lower a spec's threshold below the default.

    `aoi` is optional; when omitted, no spatial filter applies. Surviving
    detections are renumbered sequentially starting at 0 in input order.
    """

    explicit: set[str] = set()
    if specs is not None:
        explicit = {
            spec.name for spec in specs if spec.min_confidence is not None
        }

    kept: list[Detection] = []
    for det in detections:
        if det.source_model not in explicit and det.confidence < min_confidence:
            continue
        if aoi is not None and not (
            aoi.covers(det.centroid) or aoi.intersects(det.centroid)
        ):
            continue
        kept.append(det)

    return [replace(d, id=i) for i, d in enumerate(kept)]
