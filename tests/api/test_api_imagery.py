"""Tests for imagery API endpoints."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from api.main import app


@pytest.mark.asyncio
async def test_upload_imagery(tmp_path):
    with patch("api.routers.imagery.settings") as mock_settings:
        mock_settings.upload_dir = tmp_path

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/imagery/upload",
                files={"file": ("test.tif", b"fake tiff content", "image/tiff")},
            )

    assert response.status_code == 200
    data = response.json()
    assert "file_id" in data
    assert data["filename"] == "test.tif"
    assert data["size_bytes"] == len(b"fake tiff content")


@pytest.mark.asyncio
async def test_upload_invalid_extension(tmp_path):
    with patch("api.routers.imagery.settings") as mock_settings:
        mock_settings.upload_dir = tmp_path

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/imagery/upload",
                files={
                    "file": ("malware.exe", b"bad content", "application/octet-stream")
                },
            )

    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


@pytest.mark.asyncio
async def test_upload_no_extension(tmp_path):
    with patch("api.routers.imagery.settings") as mock_settings:
        mock_settings.upload_dir = tmp_path

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/imagery/upload",
                files={"file": ("noext", b"content", "application/octet-stream")},
            )

    assert response.status_code == 400


@pytest.mark.asyncio
async def test_stac_search():
    mock_item = MagicMock()
    mock_item.id = "S2A_TEST"
    mock_item.datetime = None
    mock_item.bbox = [10.0, 49.0, 10.1, 49.1]
    mock_item.assets = {"visual": MagicMock(), "data": MagicMock()}

    with patch("api.routers.imagery.StacService") as MockStac:
        instance = MockStac.return_value
        instance.search = AsyncMock(return_value=[mock_item])

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/imagery/stac/search",
                content="[10.0, 49.0, 10.1, 49.1]",
                headers={"content-type": "application/json"},
            )

    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 1
    assert data["items"][0]["id"] == "S2A_TEST"


@pytest.mark.asyncio
async def test_stac_download_not_found():
    with patch("api.routers.imagery.StacService") as MockStac:
        instance = MockStac.return_value
        instance.search = AsyncMock(return_value=[])

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/imagery/stac/download",
                params={"item_id": "nonexistent"},
                content="[10.0, 49.0, 10.1, 49.1]",
                headers={"content-type": "application/json"},
            )

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_stac_download_success(tmp_path):
    mock_item = MagicMock()
    mock_item.id = "S2A_TARGET"

    with patch("api.routers.imagery.StacService") as MockStac:
        instance = MockStac.return_value
        instance.search = AsyncMock(return_value=[mock_item])
        instance.download = AsyncMock(return_value=tmp_path / "downloaded.tif")

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/imagery/stac/download",
                params={"item_id": "S2A_TARGET"},
                content="[10.0, 49.0, 10.1, 49.1]",
                headers={"content-type": "application/json"},
            )

    assert response.status_code == 200
    data = response.json()
    assert data["item_id"] == "S2A_TARGET"
    assert "path" in data
