"""StacService -- search and download satellite imagery via STAC API."""

import asyncio
from pathlib import Path

import httpx
from pystac import Item
from pystac_client import Client

from core.config import settings


class StacService:
    """Searches STAC catalogs and downloads COG assets."""

    def __init__(self, api_url: str | None = None) -> None:
        self.api_url = api_url or settings.stac_api_url

    async def search(
        self,
        bbox: tuple[float, float, float, float],
        datetime_range: str,
        collection: str = "sentinel-2-l2a",
    ) -> list[Item]:
        """Search STAC API for items matching criteria.

        Args:
            bbox: Bounding box (west, south, east, north).
            datetime_range: ISO 8601 date range (e.g. "2024-01-01/2024-06-01").
            collection: STAC collection ID.

        Returns:
            List of matching STAC items, sorted newest first.

        Raises:
            RuntimeError: If no items match the query.
        """

        def _blocking_search() -> list[Item]:
            client = Client.open(self.api_url)
            search = client.search(
                collections=[collection],
                bbox=bbox,
                datetime=datetime_range,
            )
            return list(search.items())

        items = await asyncio.to_thread(_blocking_search)

        if not items:
            raise RuntimeError(
                f"No items found for {collection} in bbox={bbox}, "
                f"datetime={datetime_range}"
            )

        items.sort(key=lambda i: i.datetime or "", reverse=True)
        return items

    async def download(
        self,
        item: Item,
        output_dir: str | Path,
        asset_key: str = "visual",
    ) -> Path:
        """Download a COG asset from a STAC item.

        Args:
            item: A STAC Item with assets.
            output_dir: Directory to save the downloaded file.
            asset_key: Asset key to download (default: "visual").

        Returns:
            Path to the downloaded file.

        Raises:
            ValueError: If the asset key is not found in the item.
        """
        if asset_key not in item.assets:
            available = list(item.assets.keys())
            raise ValueError(
                f"Asset '{asset_key}' not found in item '{item.id}'. "
                f"Available: {available}"
            )

        url = item.assets[asset_key].href
        output_path = Path(output_dir) / f"{item.id}_{asset_key}.tif"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        async with httpx.AsyncClient() as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                with open(output_path, "wb") as f:
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        f.write(chunk)

        return output_path
