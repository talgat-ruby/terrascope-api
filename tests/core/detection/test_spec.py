import pytest

from core.detection.spec import DetectorSpec


def test_from_dict_minimal():
    spec = DetectorSpec.from_dict({"name": "yolov8n-sahi"})
    assert spec.name == "yolov8n-sahi"
    assert spec.classes is None
    assert spec.min_confidence is None
    assert spec.kwargs == {}


def test_from_dict_full():
    spec = DetectorSpec.from_dict(
        {
            "name": "segformer-landscape",
            "classes": ["building", "road"],
            "min_confidence": 0.4,
            "kwargs": {"max_dim": 512},
        }
    )
    assert spec.classes == ("building", "road")
    assert spec.min_confidence == 0.4
    assert spec.kwargs == {"max_dim": 512}


def test_from_dict_requires_name():
    with pytest.raises(ValueError, match="name"):
        DetectorSpec.from_dict({"classes": ["car"]})


def test_from_dict_rejects_bad_classes():
    with pytest.raises(ValueError, match="classes"):
        DetectorSpec.from_dict({"name": "x", "classes": "car"})


def test_from_dict_validates_confidence_range():
    with pytest.raises(ValueError, match="min_confidence"):
        DetectorSpec.from_dict({"name": "x", "min_confidence": 1.5})


def test_list_from_config_requires_detectors():
    with pytest.raises(ValueError, match="detectors"):
        DetectorSpec.list_from_config({})


def test_list_from_config_rejects_empty_list():
    with pytest.raises(ValueError, match="detectors"):
        DetectorSpec.list_from_config({"detectors": []})


def test_list_from_config_parses_multiple():
    specs = DetectorSpec.list_from_config(
        {
            "detectors": [
                {"name": "a", "classes": ["x"]},
                {"name": "b", "classes": ["y"], "min_confidence": 0.3},
            ]
        }
    )
    assert [s.name for s in specs] == ["a", "b"]
    assert specs[0].classes == ("x",)
    assert specs[1].min_confidence == 0.3


def test_list_from_config_single_accept_all_ok():
    specs = DetectorSpec.list_from_config({"detectors": [{"name": "a"}]})
    assert specs[0].classes is None


def test_list_from_config_rejects_overlapping_classes():
    with pytest.raises(ValueError, match="claimed by both"):
        DetectorSpec.list_from_config(
            {
                "detectors": [
                    {"name": "a", "classes": ["car", "ship"]},
                    {"name": "b", "classes": ["ship", "plane"]},
                ]
            }
        )


def test_list_from_config_rejects_accept_all_with_constrained():
    with pytest.raises(ValueError, match="accept all"):
        DetectorSpec.list_from_config(
            {
                "detectors": [
                    {"name": "a"},
                    {"name": "b", "classes": ["car"]},
                ]
            }
        )
