from shapely.geometry import box

from core.detection import filter_detections
from core.detection.types import Detection


def _det(
    id: int = 0,
    confidence: float = 0.8,
    bounds: tuple[float, float, float, float] = (0.1, 0.1, 0.2, 0.2),
    class_name: str = "car",
) -> Detection:
    return Detection(
        id=id,
        class_name=class_name,
        confidence=confidence,
        bbox=bounds,
        pixel_bbox=(0, 0, 10, 10),
        centroid=box(*bounds).centroid,
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
