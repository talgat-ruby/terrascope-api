"""Tests for health check endpoints."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from api.main import app


@pytest.fixture
def mock_db():
    session = AsyncMock()

    async def _get_db():
        yield session

    return session, _get_db


@pytest.fixture
def mock_temporal_client():
    client = AsyncMock()
    client.service_client = MagicMock()
    client.service_client.check_health = AsyncMock()
    return client


@pytest.mark.asyncio
async def test_health_liveness():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_health_ready_all_ok(mock_db, mock_temporal_client):
    session, get_db_override = mock_db

    from api.dependencies import get_db, get_temporal_client

    app.dependency_overrides[get_db] = get_db_override
    app.dependency_overrides[get_temporal_client] = lambda: mock_temporal_client

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health/ready")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["checks"]["database"] == "ok"
    assert data["checks"]["temporal"] == "ok"

    app.dependency_overrides = {}


@pytest.mark.asyncio
async def test_health_ready_db_down(mock_db, mock_temporal_client):
    session, get_db_override = mock_db
    session.execute = AsyncMock(side_effect=ConnectionError("DB down"))

    from api.dependencies import get_db, get_temporal_client

    app.dependency_overrides[get_db] = get_db_override
    app.dependency_overrides[get_temporal_client] = lambda: mock_temporal_client

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health/ready")

    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "degraded"
    assert data["checks"]["database"] == "unavailable"
    assert data["checks"]["temporal"] == "ok"

    app.dependency_overrides = {}


@pytest.mark.asyncio
async def test_health_ready_temporal_down(mock_db, mock_temporal_client):
    session, get_db_override = mock_db
    mock_temporal_client.service_client.check_health = AsyncMock(
        side_effect=RuntimeError("Temporal down")
    )

    from api.dependencies import get_db, get_temporal_client

    app.dependency_overrides[get_db] = get_db_override
    app.dependency_overrides[get_temporal_client] = lambda: mock_temporal_client

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health/ready")

    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "degraded"
    assert data["checks"]["database"] == "ok"
    assert data["checks"]["temporal"] == "unavailable"

    app.dependency_overrides = {}
