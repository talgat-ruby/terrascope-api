import json

from shapely.geometry import box

from core.detection.types import Detection
from core.services.indicators import IndicatorCalculatorService


def _det(
    class_name: str = "car",
    confidence: float = 0.8,
    bounds: tuple[float, float, float, float] = (10.0, 49.0, 10.01, 49.01),
    id: int = 0,
) -> Detection:
    return Detection(
        id=id,
        class_name=class_name,
        confidence=confidence,
        bbox=bounds,
        pixel_bbox=(0, 0, 10, 10),
        centroid=box(*bounds).centroid,
    )


def test_compute_basic():
    service = IndicatorCalculatorService()
    detections = [
        _det(class_name="car", bounds=(10.0, 49.0, 10.01, 49.01)),
        _det(class_name="car", bounds=(10.02, 49.02, 10.03, 49.03)),
        _det(class_name="person", bounds=(10.0, 49.0, 10.005, 49.005)),
    ]
    zones = {"zone_a": box(9.99, 48.99, 10.05, 49.05)}

    results = service.compute(detections, zones)
    assert len(results) == 2
    class_names = {r.class_name for r in results}
    assert class_names == {"car", "person"}

    car = next(r for r in results if r.class_name == "car")
    assert car.count == 2
    assert car.total_area_m2 > 0
    assert car.density_per_km2 > 0


def test_compute_detection_outside_zone():
    service = IndicatorCalculatorService()
    detections = [_det(bounds=(20.0, 60.0, 20.01, 60.01))]
    zones = {"zone_a": box(10.0, 49.0, 10.1, 49.1)}
    assert service.compute(detections, zones) == []


def test_compute_empty_detections():
    service = IndicatorCalculatorService()
    zones = {"zone_a": box(10.0, 49.0, 10.1, 49.1)}
    assert service.compute([], zones) == []


def test_export_csv(tmp_path):
    service = IndicatorCalculatorService()
    detections = [_det(bounds=(10.0, 49.0, 10.01, 49.01))]
    zones = {"zone_a": box(9.99, 48.99, 10.05, 49.05)}
    indicators = service.compute(detections, zones)

    path = service.export_csv(indicators, tmp_path / "indicators.csv")
    assert path.exists()
    assert "zone_a" in path.read_text()


def test_export_json(tmp_path):
    service = IndicatorCalculatorService()
    detections = [_det(bounds=(10.0, 49.0, 10.01, 49.01))]
    zones = {"zone_a": box(9.99, 48.99, 10.05, 49.05)}
    indicators = service.compute(detections, zones)
    path = service.export_json(indicators, tmp_path / "indicators.json")
    data = json.loads(path.read_text())
    assert len(data) == 1
