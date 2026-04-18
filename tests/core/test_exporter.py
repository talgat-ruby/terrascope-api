import geopandas as gpd
import pytest
from shapely.geometry import box

from core.services.detector import RawDetection
from core.services.exporter import GISExporterService


def _det(
    class_name: str = "building",
    confidence: int = 80,
    bounds: tuple[float, float, float, float] = (10.0, 49.0, 10.01, 49.01),
    source: str = "test",
) -> RawDetection:
    return RawDetection(
        class_name=class_name,
        confidence=confidence,
        geometry=box(*bounds),
        source=source,
    )


@pytest.fixture
def service():
    return GISExporterService()


@pytest.fixture
def sample_detections():
    return [
        _det(class_name="building", confidence=90, bounds=(10.0, 49.0, 10.01, 49.01)),
        _det(class_name="vegetation", confidence=75, bounds=(10.02, 49.02, 10.04, 49.04)),
        _det(class_name="road", confidence=60, bounds=(10.05, 49.05, 10.06, 49.06)),
    ]


def test_to_geodataframe(service, sample_detections):
    gdf = service.to_geodataframe(sample_detections)
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert len(gdf) == 3
    assert str(gdf.crs) == "EPSG:4326"
    assert "class_name" in gdf.columns
    assert "confidence" in gdf.columns
    assert "source" in gdf.columns
    assert "area_m2" in gdf.columns
    assert "geometry" in gdf.columns


def test_to_geodataframe_computes_area(service, sample_detections):
    gdf = service.to_geodataframe(sample_detections)
    assert all(gdf["area_m2"] > 0)


def test_export_geojson(service, sample_detections, tmp_path):
    gdf = service.to_geodataframe(sample_detections)
    path = service.export_geojson(gdf, tmp_path / "out.geojson")
    assert path.exists()
    read_back = gpd.read_file(path)
    assert len(read_back) == 3
    assert "class_name" in read_back.columns


def test_export_geopackage(service, sample_detections, tmp_path):
    gdf = service.to_geodataframe(sample_detections)
    path = service.export_geopackage(gdf, tmp_path / "out.gpkg")
    assert path.exists()
    read_back = gpd.read_file(path)
    assert len(read_back) == 3


def test_export_shapefile(service, sample_detections, tmp_path):
    gdf = service.to_geodataframe(sample_detections)
    path = service.export_shapefile(gdf, tmp_path / "out.shp")
    assert path.exists()
    read_back = gpd.read_file(path)
    assert len(read_back) == 3


def test_export_geojson_wgs84(service, tmp_path):
    """GeoJSON output should always be EPSG:4326 regardless of input CRS."""
    detections = [_det(bounds=(10.0, 49.0, 10.01, 49.01))]
    # Create GDF in a different CRS
    gdf = service.to_geodataframe(detections, crs="EPSG:4326")
    gdf = gdf.to_crs("EPSG:3857")

    path = service.export_geojson(gdf, tmp_path / "out.geojson")
    read_back = gpd.read_file(path)
    assert str(read_back.crs) == "EPSG:4326"


def test_export_all(service, sample_detections, tmp_path):
    paths = service.export_all(sample_detections, output_dir=tmp_path / "export")
    assert "geojson" in paths
    assert "gpkg" in paths
    assert "shp" in paths
    assert all(p.exists() for p in paths.values())


def test_export_empty_detections(service, tmp_path):
    gdf = service.to_geodataframe([])
    assert len(gdf) == 0
    assert isinstance(gdf, gpd.GeoDataFrame)

    path = service.export_geojson(gdf, tmp_path / "empty.geojson")
    assert path.exists()
