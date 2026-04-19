"""Tests for processing API endpoints."""

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from api.main import app
from core.models.processing import JobStatus, ProcessingJob


def _make_job(
    status: JobStatus = JobStatus.PENDING,
) -> ProcessingJob:
    return ProcessingJob(
        id=uuid.uuid4(),
        status=status,
        input_path="/data/test.tif",
        config={
            "aoi": {
                "type": "Polygon",
                "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
            }
        },
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


@pytest.fixture
def mock_db():
    session = AsyncMock()
    session.add = MagicMock()
    session.add_all = MagicMock()

    async def _get_db():
        yield session

    return session, _get_db


@pytest.fixture
def mock_temporal_client():
    client = AsyncMock()
    client.start_workflow = AsyncMock()
    return client


@pytest.fixture
def _override_deps(mock_db, mock_temporal_client):
    """Override both DB and Temporal dependencies for all tests that need them."""
    _, get_db_override = mock_db
    from api.dependencies import get_db, get_temporal_client

    app.dependency_overrides[get_db] = get_db_override
    app.dependency_overrides[get_temporal_client] = lambda: mock_temporal_client
    yield
    app.dependency_overrides = {}


@pytest.mark.asyncio
@pytest.mark.usefixtures("_override_deps")
async def test_start_processing(mock_db, mock_temporal_client):
    session, _ = mock_db

    job = _make_job()
    session.refresh = AsyncMock(side_effect=lambda j: setattr(j, "id", job.id))

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/processing/start",
            json={
                "input_path": "/data/test.tif",
                "aoi": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
                },
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "pending"
    mock_temporal_client.start_workflow.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.usefixtures("_override_deps")
async def test_retry_failed_job(mock_db, mock_temporal_client):
    session, _ = mock_db
    job = _make_job(status=JobStatus.FAILED)
    job.error_message = "Something went wrong"

    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = job
    session.execute = AsyncMock(return_value=result_mock)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(f"/processing/{job.id}/retry")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "retry_scheduled"
    mock_temporal_client.start_workflow.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.usefixtures("_override_deps")
async def test_retry_non_failed_job_returns_400(mock_db):
    session, _ = mock_db
    job = _make_job(status=JobStatus.COMPLETED)

    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = job
    session.execute = AsyncMock(return_value=result_mock)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(f"/processing/{job.id}/retry")

    assert response.status_code == 400
    assert "failed" in response.json()["detail"].lower()


@pytest.mark.asyncio
@pytest.mark.usefixtures("_override_deps")
async def test_get_status(mock_db):
    session, _ = mock_db
    job = _make_job(status=JobStatus.DETECTING)
    job.current_step = "detect_objects"

    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = job
    session.execute = AsyncMock(return_value=result_mock)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(f"/processing/{job.id}/status")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "detecting"
    assert data["current_step"] == "detect_objects"


@pytest.mark.asyncio
@pytest.mark.usefixtures("_override_deps")
async def test_get_status_not_found(mock_db):
    session, _ = mock_db

    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = None
    session.execute = AsyncMock(return_value=result_mock)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(f"/processing/{uuid.uuid4()}/status")

    assert response.status_code == 404


@pytest.mark.asyncio
@pytest.mark.usefixtures("_override_deps")
async def test_start_processing_temporal_failure(mock_db, mock_temporal_client):
    """When Temporal workflow start fails, job should be marked FAILED."""
    session, _ = mock_db

    job = _make_job()
    session.refresh = AsyncMock(side_effect=lambda j: setattr(j, "id", job.id))
    mock_temporal_client.start_workflow = AsyncMock(
        side_effect=RuntimeError("Temporal unavailable")
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/processing/start",
            json={
                "input_path": "/data/test.tif",
                "aoi": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
                },
            },
        )

    assert response.status_code == 500
    assert "workflow" in response.json()["detail"].lower()
