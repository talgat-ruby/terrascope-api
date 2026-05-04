"""Shared test fixtures and factories."""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import box

from core.detection.types import Detection


def make_detection(
    id: int = 0,
    class_name: str = "car",
    confidence: float = 0.85,
    bounds: tuple[float, float, float, float] = (0.1, 0.1, 0.2, 0.2),
    pixel_bbox: tuple[int, int, int, int] = (10, 10, 20, 20),
    source_model: str = "test",
) -> Detection:
    """Build a domain Detection for tests."""
    return Detection(
        id=id,
        class_name=class_name,
        confidence=confidence,
        bbox=bounds,
        pixel_bbox=pixel_bbox,
        centroid=box(*bounds).centroid,
        source_model=source_model,
    )


@pytest.fixture
def sample_polygon():
    return box(10.0, 49.0, 10.01, 49.01)


@pytest.fixture(scope="session")
def synthetic_geotiff(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("geotiff") / "test.tif"
    transform = from_bounds(10.0, 49.9, 10.1, 50.0, 100, 100)
    data = np.random.default_rng(42).random((3, 100, 100), dtype=np.float32) * 0.2
    data[:, 20:40, 20:40] = 0.9
    data[:, 60:80, 60:80] = 0.7

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
    return path


@pytest.fixture
def sample_aoi():
    return box(10.01, 49.91, 10.09, 49.99)
