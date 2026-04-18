import json

from shapely.geometry import box

from core.services.detector import RawDetection
from core.services.indicators import IndicatorCalculatorService


def _det(
    class_name: str = "building",
    confidence: int = 80,
    bounds: tuple[float, float, float, float] = (10.0, 49.0, 10.01, 49.01),
) -> RawDetection:
    return RawDetection(
        class_name=class_name,
        confidence=confidence,
        geometry=box(*bounds),
        source="test",
    )


def test_compute_basic():
    """Detections intersecting a zone should produce indicators."""
    service = IndicatorCalculatorService()
    detections = [
        _det(class_name="building", bounds=(10.0, 49.0, 10.01, 49.01)),
        _det(class_name="building", bounds=(10.02, 49.02, 10.03, 49.03)),
        _det(class_name="vegetation", bounds=(10.0, 49.0, 10.005, 49.005)),
    ]
    zones = {"zone_a": box(9.99, 48.99, 10.05, 49.05)}

    results = service.compute(detections, zones)
    assert len(results) == 2  # building + vegetation
    class_names = {r.class_name for r in results}
    assert class_names == {"building", "vegetation"}

    building = next(r for r in results if r.class_name == "building")
    assert building.count == 2
    assert building.total_area_m2 > 0
    assert building.density_per_km2 > 0


def test_compute_detection_outside_zone():
    """Detections outside the zone should not be counted."""
    service = IndicatorCalculatorService()
    detections = [_det(bounds=(20.0, 60.0, 20.01, 60.01))]
    zones = {"zone_a": box(10.0, 49.0, 10.1, 49.1)}

    results = service.compute(detections, zones)
    assert len(results) == 0


def test_compute_multiple_zones():
    """Each zone should get its own indicators."""
    service = IndicatorCalculatorService()
    detections = [
        _det(class_name="building", bounds=(10.0, 49.0, 10.01, 49.01)),
        _det(class_name="building", bounds=(11.0, 50.0, 11.01, 50.01)),
    ]
    zones = {
        "zone_a": box(9.99, 48.99, 10.05, 49.05),
        "zone_b": box(10.99, 49.99, 11.05, 50.05),
    }

    results = service.compute(detections, zones)
    zone_ids = {r.zone_id for r in results}
    assert zone_ids == {"zone_a", "zone_b"}
    assert all(r.count == 1 for r in results)


def test_compute_partial_intersection():
    """Detection partially inside zone should be clipped, area reflects clipped portion."""
    service = IndicatorCalculatorService()
    # Detection spans from 10.0 to 10.02, zone ends at 10.01
    detections = [_det(class_name="building", bounds=(10.0, 49.0, 10.02, 49.01))]
    zones = {"zone_a": box(10.0, 49.0, 10.01, 49.01)}

    results = service.compute(detections, zones)
    assert len(results) == 1
    # Area should be roughly half of the full detection
    full_det = _det(class_name="building", bounds=(10.0, 49.0, 10.02, 49.01))
    from pyproj import Geod

    geod = Geod(ellps="WGS84")
    full_area = abs(geod.geometry_area_perimeter(full_det.geometry)[0])
    assert results[0].total_area_m2 < full_area


def test_compute_empty_detections():
    """No detections should produce no indicators."""
    service = IndicatorCalculatorService()
    zones = {"zone_a": box(10.0, 49.0, 10.1, 49.1)}
    results = service.compute([], zones)
    assert len(results) == 0


def test_export_csv(tmp_path):
    service = IndicatorCalculatorService()
    detections = [_det(bounds=(10.0, 49.0, 10.01, 49.01))]
    zones = {"zone_a": box(9.99, 48.99, 10.05, 49.05)}
    indicators = service.compute(detections, zones)

    path = service.export_csv(indicators, tmp_path / "indicators.csv")
    assert path.exists()
    content = path.read_text()
    assert "zone_id" in content
    assert "zone_a" in content


def test_export_json(tmp_path):
    service = IndicatorCalculatorService()
    detections = [_det(bounds=(10.0, 49.0, 10.01, 49.01))]
    zones = {"zone_a": box(9.99, 48.99, 10.05, 49.05)}
    indicators = service.compute(detections, zones)

    path = service.export_json(indicators, tmp_path / "indicators.json")
    assert path.exists()
    data = json.loads(path.read_text())
    assert len(data) == 1
    assert data[0]["zone_id"] == "zone_a"


def test_generate_summary_table():
    service = IndicatorCalculatorService()
    detections = [
        _det(class_name="building", bounds=(10.0, 49.0, 10.01, 49.01)),
        _det(class_name="vegetation", bounds=(10.0, 49.0, 10.005, 49.005)),
    ]
    zones = {"zone_a": box(9.99, 48.99, 10.05, 49.05)}
    indicators = service.compute(detections, zones)

    summary = service.generate_summary_table(indicators)
    assert len(summary) == 1
    row = summary[0]
    assert row["zone_id"] == "zone_a"
    assert row["total_detections"] == 2
    assert "building_count" in row
    assert "vegetation_count" in row
