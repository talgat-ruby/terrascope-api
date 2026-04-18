import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import box

from core.services.imagery import ImageryLoaderService


@pytest.fixture
def service():
    return ImageryLoaderService()


@pytest.fixture
def synthetic_geotiff(tmp_path):
    """Create a 100x100, 3-band GeoTIFF with CRS=EPSG:4326."""
    path = tmp_path / "test.tif"
    transform = from_bounds(10.0, 49.9, 10.1, 50.0, 100, 100)
    data = np.random.rand(3, 100, 100).astype(np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=100,
        width=100,
        count=3,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)
    return path, data, transform


@pytest.fixture
def no_crs_geotiff(tmp_path):
    """Create a GeoTIFF without CRS."""
    path = tmp_path / "no_crs.tif"
    data = np.random.rand(3, 50, 50).astype(np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=50,
        width=50,
        count=3,
        dtype="float32",
    ) as dst:
        dst.write(data)
    return path


@pytest.fixture
def sample_aoi():
    """AOI box that intersects the synthetic GeoTIFF."""
    return box(10.01, 49.95, 10.05, 50.0)


def test_load_valid_geotiff(service, synthetic_geotiff):
    path, _, _ = synthetic_geotiff
    dataset = service.load(path)
    try:
        assert dataset.crs is not None
        assert str(dataset.crs) == "EPSG:4326"
        assert dataset.count == 3
    finally:
        dataset.close()


def test_load_no_crs_raises(service, no_crs_geotiff):
    with pytest.raises(ValueError, match="no CRS"):
        service.load(no_crs_geotiff)


def test_load_nonexistent_file(service):
    with pytest.raises(rasterio.errors.RasterioIOError):
        service.load("/nonexistent/path.tif")


def test_clip_to_aoi(service, synthetic_geotiff, sample_aoi):
    path, original_data, _ = synthetic_geotiff
    dataset = service.load(path)
    try:
        clipped_data, clipped_transform = service.clip_to_aoi(dataset, sample_aoi)
        assert clipped_data.dtype == np.float32
        assert clipped_data.shape[0] == 3  # same band count
        assert clipped_data.shape[1] < original_data.shape[1]  # smaller height
        assert clipped_data.shape[2] < original_data.shape[2]  # smaller width
        assert clipped_transform is not None
    finally:
        dataset.close()


def test_clip_to_aoi_crs_mismatch(service, synthetic_geotiff):
    """AOI in EPSG:3857 should be reprojected and still clip correctly."""
    path, _, _ = synthetic_geotiff
    from pyproj import Transformer
    from shapely import ops

    # Create AOI in EPSG:3857
    aoi_4326 = box(10.01, 49.95, 10.05, 50.0)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    aoi_3857 = ops.transform(transformer.transform, aoi_4326)

    dataset = service.load(path)
    try:
        clipped_data, _ = service.clip_to_aoi(dataset, aoi_3857, aoi_crs="EPSG:3857")
        assert clipped_data.dtype == np.float32
        assert clipped_data.shape[0] == 3
        assert clipped_data.shape[1] > 0
        assert clipped_data.shape[2] > 0
    finally:
        dataset.close()


def test_clip_to_aoi_no_intersection(service, synthetic_geotiff):
    """AOI completely outside raster bounds should raise ValueError."""
    path, _, _ = synthetic_geotiff
    far_away_aoi = box(20.0, 60.0, 21.0, 61.0)
    dataset = service.load(path)
    try:
        with pytest.raises(ValueError, match="do not overlap|does not intersect"):
            service.clip_to_aoi(dataset, far_away_aoi)
    finally:
        dataset.close()


def test_get_metadata(service, synthetic_geotiff):
    path, _, transform = synthetic_geotiff
    dataset = service.load(path)
    try:
        meta = service.get_metadata(dataset)
        assert meta["band_count"] == 3
        assert meta["crs"] == "EPSG:4326"
        assert "left" in meta["bounds"]
        assert "bottom" in meta["bounds"]
        assert "right" in meta["bounds"]
        assert "top" in meta["bounds"]
        assert pytest.approx(meta["bounds"]["left"], abs=1e-6) == 10.0
        assert pytest.approx(meta["bounds"]["top"], abs=1e-6) == 50.0
        assert len(meta["resolution"]) == 2
        assert meta["resolution"][0] > 0
    finally:
        dataset.close()
