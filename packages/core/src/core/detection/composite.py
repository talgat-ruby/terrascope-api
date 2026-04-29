"""CompositeDetector -- run multiple Detectors and merge their output.

Each child Detector is invoked independently against the same Raster; the
returned `Detection` lists are concatenated and renumbered with sequential
ids. Useful for combining an object detector (e.g., DOTA-trained YOLO) with
a landscape segmenter (e.g., SegFormer-ADE20K) so a single run yields both
object bboxes and land-cover regions.
"""

from __future__ import annotations

from dataclasses import replace

from core.detection.types import Detection, Detector, Raster


class CompositeDetector:
    """Pluggable Detector that composes multiple child Detectors."""

    def __init__(self, children: list[Detector], name: str = "composite") -> None:
        if not children:
            raise ValueError("CompositeDetector requires at least one child")
        self.children = children
        self.name = name

    def detect(self, raster: Raster) -> list[Detection]:
        merged: list[Detection] = []
        for child in self.children:
            merged.extend(child.detect(raster))
        return [replace(d, id=i) for i, d in enumerate(merged)]
