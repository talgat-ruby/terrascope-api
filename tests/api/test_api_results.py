"""Tests for results API endpoints."""

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from api.main import app
from core.models.processing import JobStatus, ProcessingJob


def _make_completed_job(export_formats: dict | None = None) -> ProcessingJob:
    return ProcessingJob(
        id=uuid.uuid4(),
        status=JobStatus.COMPLETED,
        input_path="/data/test.tif",
        config={
            "aoi": {
                "type": "Polygon",
                "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
            }
        },
        checkpoint_data={
            "export": {
                "formats": export_formats or {},
            }
        },
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
    )


@pytest.fixture
def mock_db():
    session = AsyncMock()

    async def _get_db():
        yield session

    return session, _get_db


@pytest.mark.asyncio
async def test_download_geojson(mock_db, tmp_path):
    session, get_db_override = mock_db

    # Create a real file on disk
    geojson_path = tmp_path / "detections.geojson"
    geojson_path.write_text('{"type": "FeatureCollection", "features": []}')

    job = _make_completed_job(
        export_formats={"geojson": str(geojson_path), "gpkg": "/fake/path.gpkg"}
    )

    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = job
    session.execute = AsyncMock(return_value=result_mock)

    from api.dependencies import get_db

    app.dependency_overrides[get_db] = get_db_override

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(f"/results/{job.id}/download?format=geojson")

    assert response.status_code == 200
    assert "geo+json" in response.headers.get("content-type", "")

    app.dependency_overrides = {}


@pytest.mark.asyncio
async def test_download_incomplete_job(mock_db):
    session, get_db_override = mock_db

    job = ProcessingJob(
        id=uuid.uuid4(),
        status=JobStatus.DETECTING,
        input_path="/data/test.tif",
        config={},
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )

    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = job
    session.execute = AsyncMock(return_value=result_mock)

    from api.dependencies import get_db

    app.dependency_overrides[get_db] = get_db_override

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(f"/results/{job.id}/download?format=geojson")

    assert response.status_code == 400
    assert "not completed" in response.json()["detail"].lower()

    app.dependency_overrides = {}


@pytest.mark.asyncio
async def test_download_missing_file(mock_db):
    session, get_db_override = mock_db

    job = _make_completed_job(
        export_formats={"geojson": "/nonexistent/path/detections.geojson"}
    )

    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = job
    session.execute = AsyncMock(return_value=result_mock)

    from api.dependencies import get_db

    app.dependency_overrides[get_db] = get_db_override

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(f"/results/{job.id}/download?format=geojson")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()

    app.dependency_overrides = {}


@pytest.mark.asyncio
async def test_download_not_found_job(mock_db):
    session, get_db_override = mock_db

    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = None
    session.execute = AsyncMock(return_value=result_mock)

    from api.dependencies import get_db

    app.dependency_overrides[get_db] = get_db_override

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(f"/results/{uuid.uuid4()}/download?format=geojson")

    assert response.status_code == 404

    app.dependency_overrides = {}
