from shapely.geometry import box

from core.services.detector import RawDetection
from core.services.postprocessor import PostprocessingConfig, PostprocessorService


def _det(
    class_name: str = "building",
    confidence: int = 80,
    bounds: tuple[float, float, float, float] = (0, 0, 1, 1),
    source: str = "test",
) -> RawDetection:
    """Helper to create a RawDetection with a box geometry."""
    return RawDetection(
        class_name=class_name,
        confidence=confidence,
        geometry=box(*bounds),
        source=source,
    )


def test_merge_tile_detections():
    service = PostprocessorService()
    tile1 = [_det(bounds=(0, 0, 1, 1)), _det(bounds=(1, 0, 2, 1))]
    tile2 = [_det(bounds=(2, 0, 3, 1))]
    merged = service.merge_tile_detections([tile1, tile2])
    assert len(merged) == 3


def test_nms_suppresses_overlapping():
    """Overlapping detections of same class: lower confidence is suppressed."""
    service = PostprocessorService()
    high = _det(confidence=90, bounds=(0, 0, 1, 1))
    low = _det(confidence=60, bounds=(0.1, 0.1, 1.1, 1.1))  # ~81% IoU
    result = service.apply_nms([high, low], iou_threshold=0.5)
    assert len(result) == 1
    assert result[0].confidence == 90


def test_nms_keeps_non_overlapping():
    """Non-overlapping detections should both be kept."""
    service = PostprocessorService()
    a = _det(confidence=90, bounds=(0, 0, 1, 1))
    b = _det(confidence=80, bounds=(5, 5, 6, 6))
    result = service.apply_nms([a, b], iou_threshold=0.5)
    assert len(result) == 2


def test_nms_different_classes_independent():
    """NMS should not suppress across different classes."""
    service = PostprocessorService()
    building = _det(class_name="building", confidence=90, bounds=(0, 0, 1, 1))
    vegetation = _det(class_name="vegetation", confidence=80, bounds=(0, 0, 1, 1))
    result = service.apply_nms([building, vegetation], iou_threshold=0.5)
    assert len(result) == 2


def test_filter_by_confidence():
    service = PostprocessorService()
    detections = [_det(confidence=30), _det(confidence=60), _det(confidence=90)]
    result = service.filter_by_confidence(detections, threshold=50)
    assert len(result) == 2
    assert all(d.confidence >= 50 for d in result)


def test_filter_by_confidence_exact_threshold():
    """Detection at exactly the threshold should be kept."""
    service = PostprocessorService()
    result = service.filter_by_confidence([_det(confidence=50)], threshold=50)
    assert len(result) == 1


def test_filter_by_shape_removes_invalid():
    """Invalid and degenerate geometries should be removed."""
    service = PostprocessorService()
    valid = _det(bounds=(0, 0, 1, 1))
    # Create a zero-width geometry (degenerate line)
    degenerate = _det(bounds=(0, 0, 0, 1))
    result = service.filter_by_shape([valid, degenerate])
    assert len(result) == 1


def test_filter_by_shape_removes_slivers():
    """Very thin slivers (aspect ratio > 100) should be removed."""
    service = PostprocessorService()
    normal = _det(bounds=(0, 0, 1, 1))
    sliver = _det(bounds=(0, 0, 0.001, 1))  # aspect = 1000
    result = service.filter_by_shape([normal, sliver])
    assert len(result) == 1


def test_simplify_geometries():
    service = PostprocessorService()
    detections = [_det(bounds=(0, 0, 1, 1))]
    result = service.simplify_geometries(detections, tolerance=0.01)
    assert len(result) == 1
    assert result[0].geometry.is_valid


def test_clip_to_aoi_fully_inside():
    """Detection fully inside AOI should be unchanged."""
    service = PostprocessorService()
    aoi = box(0, 0, 10, 10)
    det = _det(bounds=(1, 1, 2, 2))
    result = service.clip_to_aoi([det], aoi)
    assert len(result) == 1
    assert result[0].geometry.area == det.geometry.area


def test_clip_to_aoi_partially_outside():
    """Detection partially outside should be clipped."""
    service = PostprocessorService()
    aoi = box(0, 0, 1, 1)
    det = _det(bounds=(0.5, 0.5, 1.5, 1.5))
    result = service.clip_to_aoi([det], aoi)
    assert len(result) == 1
    assert result[0].geometry.area < det.geometry.area


def test_clip_to_aoi_fully_outside():
    """Detection fully outside AOI should be removed."""
    service = PostprocessorService()
    aoi = box(0, 0, 1, 1)
    det = _det(bounds=(5, 5, 6, 6))
    result = service.clip_to_aoi([det], aoi)
    assert len(result) == 0


def test_run_full_pipeline():
    """Full pipeline should apply all steps and return stats."""
    service = PostprocessorService()
    config = PostprocessingConfig(
        iou_threshold=0.5,
        confidence_threshold=50,
        simplify_tolerance=0.001,
    )
    aoi = box(0, 0, 10, 10)

    detections = [
        _det(confidence=90, bounds=(0, 0, 1, 1)),
        _det(confidence=30, bounds=(2, 2, 3, 3)),  # below confidence
        _det(confidence=60, bounds=(0.1, 0.1, 1.1, 1.1)),  # NMS suppressed
        _det(confidence=80, bounds=(5, 5, 6, 6)),
        _det(confidence=70, bounds=(20, 20, 21, 21)),  # outside AOI
    ]

    result, stats = service.run(detections, config, aoi=aoi)

    assert stats["input_count"] == 5
    assert stats["output_count"] == len(result)
    assert stats["output_count"] < stats["input_count"]
    assert all(d.confidence >= 50 for d in result)


def test_run_without_aoi():
    """Pipeline should work without AOI clipping."""
    service = PostprocessorService()
    detections = [_det(confidence=80, bounds=(0, 0, 1, 1))]
    result, stats = service.run(detections)
    assert "after_clip" not in stats
    assert stats["output_count"] == 1


def test_run_empty_detections():
    """Empty input should return empty output."""
    service = PostprocessorService()
    result, stats = service.run([])
    assert len(result) == 0
    assert stats["input_count"] == 0
    assert stats["output_count"] == 0
