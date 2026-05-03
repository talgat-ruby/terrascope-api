import pytest

from core2 import ProcessSpec


def test_from_dict_minimal():
    spec = ProcessSpec.from_dict({"name": "msft-buildings"})
    assert spec.name == "msft-buildings"
    assert spec.classes is None
    assert spec.min_confidence is None
    assert spec.kwargs == {}


def test_from_dict_full():
    spec = ProcessSpec.from_dict(
        {
            "name": "yolov8-satellite-vehicle",
            "classes": ["car"],
            "min_confidence": 0.4,
            "kwargs": {"device": "cpu"},
        }
    )
    assert spec.classes == ("car",)
    assert spec.min_confidence == 0.4
    assert spec.kwargs == {"device": "cpu"}


def test_from_dict_requires_name():
    with pytest.raises(ValueError, match="name"):
        ProcessSpec.from_dict({"classes": ["car"]})


def test_from_dict_validates_confidence_range():
    with pytest.raises(ValueError, match="min_confidence"):
        ProcessSpec.from_dict({"name": "x", "min_confidence": 1.5})


def test_list_from_config_requires_processes():
    with pytest.raises(ValueError, match="processes"):
        ProcessSpec.list_from_config({})


def test_list_from_config_parses_multiple():
    specs = ProcessSpec.list_from_config(
        {
            "processes": [
                {"name": "msft-buildings"},
                {"name": "yolov8-satellite-vehicle", "classes": ["car"]},
            ]
        }
    )
    assert [s.name for s in specs] == ["msft-buildings", "yolov8-satellite-vehicle"]
