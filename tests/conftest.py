"""Shared test fixtures and factories."""

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from shapely.geometry import box

from core.models.processing import JobStatus, ProcessingJob
from core.services.detector import RawDetection


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
