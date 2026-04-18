import uuid
from pathlib import Path

from fastapi import APIRouter, UploadFile

from core.config import settings
from core.services.stac import StacService

router = APIRouter()


@router.post("/upload")
async def upload_imagery(file: UploadFile) -> dict:
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    file_id = uuid.uuid4()
    dest = settings.upload_dir / f"{file_id}_{file.filename}"
    content = await file.read()
    dest.write_bytes(content)
    return {
        "file_id": str(file_id),
        "filename": file.filename,
        "path": str(dest),
        "size_bytes": len(content),
    }


@router.post("/stac/search")
async def stac_search(
    bbox: list[float],
    datetime_range: str | None = None,
    collection: str = "sentinel-2-l2a",
) -> dict:
    stac = StacService()
    items = await stac.search(
        bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
        datetime_range=datetime_range or "",
        collection=collection,
    )
    return {
        "count": len(items),
        "items": [
            {
                "id": item.id,
                "datetime": str(item.datetime) if item.datetime else None,
                "bbox": list(item.bbox) if item.bbox else None,
                "assets": list(item.assets.keys()),
            }
            for item in items
        ],
    }


@router.post("/stac/download")
async def stac_download(
    item_id: str,
    bbox: list[float],
    datetime_range: str | None = None,
    collection: str = "sentinel-2-l2a",
    asset_key: str = "visual",
    output_dir: str | None = None,
) -> dict:
    stac = StacService()
    # Search for the item to get its full metadata
    items = await stac.search(
        bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
        datetime_range=datetime_range or "",
        collection=collection,
    )
    item = next((i for i in items if i.id == item_id), None)
    if item is None:
        return {"status": "error", "message": f"Item {item_id} not found"}

    dest = Path(output_dir) if output_dir else settings.upload_dir
    downloaded_path = await stac.download(item, dest, asset_key=asset_key)
    return {
        "item_id": item_id,
        "path": str(downloaded_path),
    }
