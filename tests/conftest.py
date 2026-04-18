"""Shared test fixtures and factories."""

import uuid
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import box

from core.models.processing import JobStatus, ProcessingJob
from core.models.tile import Tile
from core.services.detector import DetectorService, RawDetection


def make_raw_detection(
    class_name: str = "building",
    confidence: int = 85,
    bounds: tuple[float, float, float, float] = (0.1, 0.1, 0.2, 0.2),
    source: str = "test",
) -> RawDetection:
    """Create a RawDetection for testing."""
    return RawDetection(
        class_name=class_name,
        confidence=confidence,
        geometry=box(*bounds),
        source=source,
    )


def make_job(
    job_id: str | None = None,
    status: JobStatus = JobStatus.PENDING,
    config: dict | None = None,
    checkpoint_data: dict | None = None,
) -> ProcessingJob:
    """Create a ProcessingJob for testing."""
    return ProcessingJob(
        id=uuid.UUID(job_id) if job_id else uuid.uuid4(),
        status=status,
        input_path="/data/test.tif",
        config=config
        or {
            "aoi": {
                "type": "Polygon",
                "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
            },
            "aoi_crs": "EPSG:4326",
        },
        checkpoint_data=checkpoint_data,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


@pytest.fixture
def sample_polygon():
    """A small test polygon."""
    return box(10.0, 49.0, 10.01, 49.01)


@pytest.fixture
def mock_async_session():
    """AsyncSession mock with sync methods properly mocked."""
    session = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    # add/add_all are sync methods on AsyncSession;
    # using AsyncMock causes "coroutine never awaited" warnings
    session.add = MagicMock()
    session.add_all = MagicMock()
    return session


# ---------------------------------------------------------------------------
# Shared fixtures for integration tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def synthetic_geotiff(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """100x100, 3-band GeoTIFF at EPSG:4326 with simulated objects.

    Bounds: (10.0, 49.9, 10.1, 50.0).
    - Rows 20-40, cols 20-40: bright region (building-like)
    - Rows 60-80, cols 60-80: bright region (vegetation-like)
    """
    path = tmp_path_factory.mktemp("geotiff") / "test.tif"
    transform = from_bounds(10.0, 49.9, 10.1, 50.0, 100, 100)
    data = np.random.default_rng(42).random((3, 100, 100), dtype=np.float32) * 0.2

    # Paint bright object regions
    data[:, 20:40, 20:40] = 0.9  # building-like
    data[:, 60:80, 60:80] = 0.7  # vegetation-like

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
    """AOI polygon that intersects the synthetic GeoTIFF bounds."""
    return box(10.01, 49.91, 10.09, 49.99)


@pytest.fixture
def mock_detector() -> DetectorService:
    """DetectorService that uses predict_tile_with_masks with synthetic masks.

    Produces one building and one vegetation detection per tile by injecting
    high-probability regions into the mask. Exercises real _mask_to_polygons.
    """
    detector = DetectorService(device="cpu")

    _real_predict_with_masks = detector.predict_tile_with_masks

    def _fake_predict_tile(tile: Tile) -> list[RawDetection]:
        h, w = tile.data.shape[1], tile.data.shape[2]
        masks: dict[str, np.ndarray] = {
            "building": np.zeros((h, w), dtype=np.float32),
            "vegetation": np.zeros((h, w), dtype=np.float32),
        }
        # Place a high-prob block in the top-left quadrant for building
        qh, qw = h // 4, w // 4
        masks["building"][qh : qh * 2, qw : qw * 2] = 0.9
        # Place a high-prob block in the bottom-right quadrant for vegetation
        masks["vegetation"][qh * 2 : qh * 3, qw * 2 : qw * 3] = 0.8
        return _real_predict_with_masks(tile, masks, source="mock")

    detector.predict_tile = _fake_predict_tile  # type: ignore[assignment]
    return detector
