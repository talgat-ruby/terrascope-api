"""Unit tests for the SAM-RS process.

We don't load real SAM weights here — instead we stub
`SamAutomaticMaskGenerator.generate` with hand-crafted mask lists and
verify class assignment via the per-class shape rules.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from affine import Affine

from core.detection.types import Raster
from core2 import ProcessSpec, build, registered_names
from core2.processes.sam_rs import SamRsProcess


def _raster(h: int = 512, w: int = 512) -> Raster:
    return Raster(
        data=np.zeros((h, w, 3), dtype=np.uint8),
        transform=Affine.translation(0, 0) * Affine.scale(1, -1),
        crs="EPSG:4326",
        aoi_geom=None,
    )


def test_sam_rs_registered():
    assert "sam-rs" in registered_names()
    assert "sam-rs-roads" in registered_names()  # back-compat
    proc = build(ProcessSpec(name="sam-rs"))
    assert proc.name == "sam-rs"


def test_classifies_road_and_building_in_one_run():
    spec = ProcessSpec(
        name="sam-rs",
        classes=("road", "building"),
    )
    proc = SamRsProcess.from_spec(spec)

    fake_masks = [
        # Road-shaped: long, skinny.
        {"bbox": [10, 10, 200, 8], "area": 1600, "predicted_iou": 0.9},
        # Building-shaped: roughly square, mid-size.
        {"bbox": [50, 50, 60, 50], "area": 3000, "predicted_iou": 0.88},
        # Tiny: rejected by both classes.
        {"bbox": [0, 0, 30, 4], "area": 120, "predicted_iou": 0.9},
        # Huge sky-like blob: rejected by both max_area_frac caps.
        {"bbox": [0, 0, 500, 500], "area": 250_000, "predicted_iou": 0.9},
    ]

    with patch.object(SamRsProcess, "_load", lambda self: None):
        proc._generator = type("G", (), {"generate": lambda self_, _img: fake_masks})()
        dets = proc.run(_raster())

    classes = sorted(d.class_name for d in dets)
    assert classes == ["building", "road"]
    for d in dets:
        assert d.source_model == "sam-rs"


def test_unknown_class_in_spec_raises():
    spec = ProcessSpec(name="sam-rs", classes=("vegetation",))
    with pytest.raises(ValueError, match="no shape rules"):
        SamRsProcess.from_spec(spec)


def test_class_rules_can_be_overridden_via_kwargs():
    spec = ProcessSpec(
        name="sam-rs",
        classes=("vegetation",),
        kwargs={
            "class_rules": {
                "vegetation": {
                    "min_area_px": 500,
                    "max_elongation": 3.0,
                    "min_long_side_px": 30,
                    "max_area_frac": 0.9,
                }
            }
        },
    )
    proc = SamRsProcess.from_spec(spec)
    assert "vegetation" in proc.class_rules
    assert proc.class_rules["vegetation"]["min_area_px"] == 500


def test_resolve_weights_rejects_unknown_filename(tmp_path):
    spec = ProcessSpec(
        name="sam-rs",
        kwargs={"weights": "not_a_real_checkpoint.pth", "weights_dir": str(tmp_path)},
    )
    proc = SamRsProcess.from_spec(spec)
    with pytest.raises(FileNotFoundError, match="sam_vit_h_4b8939.pth"):
        proc._resolve_weights()
