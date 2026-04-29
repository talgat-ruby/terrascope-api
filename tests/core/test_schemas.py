import uuid

from core.schemas.detection import DetectionCreate, DetectionResponse
from core.schemas.export import ExportRequest
from core.schemas.indicator import IndicatorResponse
from core.schemas.processing import ProcessingRequest
from core.schemas.territory import TerritoryCreate

_POINT = {"type": "Point", "coordinates": [10.0, 49.0]}
_BBOX = {
    "type": "Polygon",
    "coordinates": [
        [[10.0, 49.0], [10.01, 49.0], [10.01, 49.01], [10.0, 49.01], [10.0, 49.0]]
    ],
}


def test_detection_create():
    d = DetectionCreate(
        class_name="car",
        confidence=0.85,
        geometry=_BBOX,
    )
    assert d.class_name == "car"
    assert d.confidence == 0.85


def test_detection_response():
    d = DetectionResponse(
        id=42,
        job_id=uuid.uuid4(),
        class_name="building",
        confidence=0.7,
        geometry=_BBOX,
        centroid=_POINT,
    )
    assert d.id == 42
    assert d.confidence == 0.7


def test_processing_request():
    req = ProcessingRequest(
        input_path="/data/image.tif",
        aoi={
            "type": "Polygon",
            "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
        },
    )
    assert req.aoi_crs == "EPSG:4326"


def test_territory_create():
    t = TerritoryCreate(
        name="Test Area",
        geometry={
            "type": "Polygon",
            "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
        },
    )
    assert t.name == "Test Area"
    assert t.crs == "EPSG:4326"


def test_export_request_defaults():
    e = ExportRequest()
    assert "geojson" in e.formats
    assert e.target_crs == "EPSG:4326"


def test_indicator_response():
    i = IndicatorResponse(
        id=uuid.uuid4(),
        job_id=uuid.uuid4(),
        zone_id=uuid.uuid4(),
        class_name="car",
        count=150,
        density_per_km2=25.0,
        total_area_m2=500000.0,
    )
    assert i.count == 150
