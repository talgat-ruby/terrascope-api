from shapely.geometry import box

from core.detection import filter_detections
from core.detection.spec import DetectorSpec
from core.detection.types import Detection


def _det(
    id: int = 0,
    confidence: float = 0.8,
    bounds: tuple[float, float, float, float] = (0.1, 0.1, 0.2, 0.2),
    class_name: str = "car",
    source_model: str = "stub",
) -> Detection:
    return Detection(
        id=id,
        class_name=class_name,
        confidence=confidence,
        bbox=bounds,
        pixel_bbox=(0, 0, 10, 10),
        centroid=box(*bounds).centroid,
        source_model=source_model,
    )


def test_drops_below_threshold():
    out = filter_detections(
        [_det(confidence=0.1), _det(confidence=0.5), _det(confidence=0.9)],
        min_confidence=0.4,
    )
    assert len(out) == 2
    assert all(d.confidence >= 0.4 for d in out)


def test_threshold_inclusive():
    out = filter_detections([_det(confidence=0.5)], min_confidence=0.5)
    assert len(out) == 1


def test_aoi_filter_drops_centroids_outside():
    aoi = box(0, 0, 1, 1)
    inside = _det(bounds=(0.1, 0.1, 0.2, 0.2))
    outside = _det(bounds=(5.0, 5.0, 5.1, 5.1))
    out = filter_detections([inside, outside], min_confidence=0.0, aoi=aoi)
    assert len(out) == 1


def test_no_aoi_keeps_all():
    out = filter_detections(
        [_det(bounds=(5, 5, 6, 6)), _det(bounds=(0, 0, 1, 1))], min_confidence=0.0
    )
    assert len(out) == 2


def test_renumbers_ids():
    out = filter_detections([_det(id=99), _det(id=42), _det(id=7)], min_confidence=0.0)
    assert [d.id for d in out] == [0, 1, 2]


def test_empty_input():
    assert filter_detections([], min_confidence=0.5) == []


def test_per_spec_min_confidence_overrides_global_default():
    permissive_spec = DetectorSpec(name="seg", min_confidence=0.1)
    out = filter_detections(
        [_det(confidence=0.15, class_name="building", source_model="seg")],
        min_confidence=0.25,
        specs=[permissive_spec],
    )
    assert len(out) == 1


def test_global_default_still_applies_when_spec_has_no_explicit_threshold():
    spec = DetectorSpec(name="seg")  # no min_confidence
    out = filter_detections(
        [_det(confidence=0.15, class_name="building", source_model="seg")],
        min_confidence=0.25,
        specs=[spec],
    )
    assert out == []


def test_specs_match_by_source_model():
    spec_seg = DetectorSpec(name="seg", min_confidence=0.1)
    spec_yolo = DetectorSpec(name="yolo")  # default applies
    seg_det = Detection(
        id=0, class_name="building", confidence=0.15,
        bbox=(0.1, 0.1, 0.2, 0.2), pixel_bbox=(0, 0, 10, 10),
        centroid=box(0.1, 0.1, 0.2, 0.2).centroid, source_model="seg",
    )
    yolo_det = Detection(
        id=1, class_name="car", confidence=0.15,
        bbox=(0.3, 0.3, 0.4, 0.4), pixel_bbox=(0, 0, 10, 10),
        centroid=box(0.3, 0.3, 0.4, 0.4).centroid, source_model="yolo",
    )
    out = filter_detections(
        [seg_det, yolo_det], min_confidence=0.25, specs=[spec_seg, spec_yolo]
    )
    assert [d.class_name for d in out] == ["building"]
