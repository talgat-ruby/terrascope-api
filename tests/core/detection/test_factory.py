import pytest

from core.detection.factory import build_detector


def test_unknown_detector_raises():
    with pytest.raises(ValueError, match="Unknown detector"):
        build_detector("does-not-exist")


def test_default_known_detectors_listed_on_error():
    with pytest.raises(ValueError, match="yolov8n-sahi"):
        build_detector("nope")


def test_known_detectors_include_landscape_and_composite():
    """Smoke check that the expected names are registered."""
    from core.detection.factory import _BUILDERS

    assert "segformer-landscape" in _BUILDERS
    assert "aerial+landscape" in _BUILDERS
