"""Tests for grounded-sam: prompt building, label mapping, end-to-end stubs."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from affine import Affine

from core.detection.types import Raster
from core2 import ProcessSpec, build, registered_names
from core2.processes.grounded_sam import GroundedSamProcess, _build_prompt


def _raster(h: int = 256, w: int = 256) -> Raster:
    return Raster(
        data=np.zeros((h, w, 3), dtype=np.uint8),
        transform=Affine.translation(0, 0) * Affine.scale(1, -1),
        crs="EPSG:4326",
        aoi_geom=None,
    )


def test_grounded_sam_registered():
    assert "grounded-sam" in registered_names()
    proc = build(ProcessSpec(name="grounded-sam", classes=("road",)))
    assert proc.name == "grounded-sam"


def test_requires_classes():
    with pytest.raises(ValueError, match="requires spec.classes"):
        GroundedSamProcess.from_spec(ProcessSpec(name="grounded-sam"))


def test_build_prompt_with_synonyms():
    prompt, label_map = _build_prompt(
        ("road", "building"),
        {"road": ["street", "highway"]},
    )
    assert prompt == "road. street. highway. building."
    assert label_map["street"] == "road"
    assert label_map["highway"] == "road"
    assert label_map["building"] == "building"


def test_run_maps_labels_to_canonical_classes():
    spec = ProcessSpec(
        name="grounded-sam",
        classes=("road", "building"),
        kwargs={"extra_text_prompts": {"road": ["street"]}},
    )
    proc = GroundedSamProcess.from_spec(spec)

    fake_boxes = [(10, 10, 50, 20), (100, 100, 130, 130)]
    fake_scores = [0.42, 0.55]
    fake_labels = ["street", "building"]
    fake_mask_boxes = [(10, 10, 50, 20), (100, 100, 130, 130)]

    with patch.object(GroundedSamProcess, "_load", lambda self: None), \
         patch.object(
             GroundedSamProcess,
             "_gdino_detect",
             lambda self, image, prompt: (fake_boxes, fake_scores, fake_labels),
         ), \
         patch.object(
             GroundedSamProcess,
             "_sam_predict_boxes",
             lambda self, boxes: fake_mask_boxes,
         ):
        proc._sam_predictor = type(
            "P", (), {"set_image": lambda self_, _img: None}
        )()
        dets = proc.run(_raster())

    classes = sorted(d.class_name for d in dets)
    assert classes == ["building", "road"]
    assert all(d.source_model == "grounded-sam" for d in dets)
    by_class = {d.class_name: d for d in dets}
    assert by_class["road"].confidence == pytest.approx(0.42)
    assert by_class["building"].confidence == pytest.approx(0.55)
