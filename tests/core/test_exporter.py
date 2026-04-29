import geopandas as gpd
import pytest
from shapely.geometry import box

from core.detection.types import Detection
from core.services.exporter import GISExporterService


def _det(
    id: int = 0,
    class_name: str = "car",
    confidence: float = 0.8,
    bounds: tuple[float, float, float, float] = (10.0, 49.0, 10.01, 49.01),
) -> Detection:
    return Detection(
        id=id,
        class_name=class_name,
        confidence=confidence,
        bbox=bounds,
        pixel_bbox=(0, 0, 10, 10),
        centroid=box(*bounds).centroid,
    )


@pytest.fixture
def service():
    return GISExporterService()


@pytest.fixture
def sample_detections():
    return [
        _det(id=0, class_name="car", confidence=0.9, bounds=(10.0, 49.0, 10.01, 49.01)),
        _det(
            id=1,
            class_name="person",
            confidence=0.75,
            bounds=(10.02, 49.02, 10.04, 49.04),
        ),
    ]


def test_to_geodataframe_columns(service, sample_detections):
    gdf = service.to_geodataframe(sample_detections)
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert len(gdf) == 2
    assert str(gdf.crs) == "EPSG:4326"
    for col in ("id", "class_name", "confidence", "geometry", "centroid_wkt"):
        assert col in gdf.columns


def test_geometry_is_polygon(service, sample_detections):
    gdf = service.to_geodataframe(sample_detections)
    assert all(g.geom_type == "Polygon" for g in gdf.geometry)


def test_export_geojson_roundtrip(service, sample_detections, tmp_path):
    path = service.export_geojson(sample_detections, tmp_path / "out.geojson")
    assert path.exists()
    read_back = gpd.read_file(path)
    assert len(read_back) == 2
    assert "class_name" in read_back.columns
    assert "centroid_wkt" in read_back.columns
    assert all(g.geom_type == "Polygon" for g in read_back.geometry)


def test_export_empty(service, tmp_path):
    path = service.export_geojson([], tmp_path / "empty.geojson")
    assert path.exists()
