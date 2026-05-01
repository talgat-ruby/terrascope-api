import pytest

from core.detection.factory import _BUILDERS, build_from_specs, build_leaf
from core.detection.spec import DetectorSpec


def test_unknown_detector_raises():
    with pytest.raises(ValueError, match="Unknown detector"):
        build_leaf(DetectorSpec(name="does-not-exist"))


def test_unknown_lists_known_in_message():
    with pytest.raises(ValueError, match="yolov8n-sahi"):
        build_leaf(DetectorSpec(name="nope"))


def test_known_leaf_detectors():
    assert "segformer-landscape" in _BUILDERS
    assert "yolov8-obb-aerial" in _BUILDERS
    assert "yolov8n-sahi" in _BUILDERS


def test_build_from_specs_requires_specs():
    with pytest.raises(ValueError, match="at least one"):
        build_from_specs([])
