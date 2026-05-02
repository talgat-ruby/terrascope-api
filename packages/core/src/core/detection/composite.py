"""CompositeDetector -- run multiple Detectors with per-detector class
allowlists and confidence thresholds, then merge their output.

Each child runs independently against the same Raster. Outputs are filtered
to the spec's class allowlist and confidence floor, stamped with provenance,
and concatenated. Canonical id renumbering is owned by `filter_detections`.
"""

from __future__ import annotations

from dataclasses import replace

from core.detection.spec import DetectorSpec
from core.detection.types import Detection, Detector, Raster


class CompositeDetector:
    """Pluggable Detector composing multiple child Detectors with specs."""

    name = "composite"

    def __init__(self, pairs: list[tuple[Detector, DetectorSpec]]) -> None:
        if not pairs:
            raise ValueError("CompositeDetector requires at least one child")
        self.pairs = pairs

    def detect(self, raster: Raster) -> list[Detection]:
        merged: list[Detection] = []
        for child, spec in self.pairs:
            allow = set(spec.classes) if spec.classes is not None else None
            min_conf = spec.min_confidence
            for det in child.detect(raster):
                if allow is not None and det.class_name not in allow:
                    continue
                if min_conf is not None and det.confidence < min_conf:
                    continue
                if det.source_model != child.name:
                    det = replace(det, source_model=child.name)
                merged.append(det)
        return merged
