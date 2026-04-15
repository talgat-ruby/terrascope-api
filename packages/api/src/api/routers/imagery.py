import uuid
from pathlib import Path

from fastapi import APIRouter, UploadFile

from core.config import settings

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
    # Stub -- will be implemented in Phase 2
    return {"status": "not_implemented", "message": "STAC search will be available in Phase 2"}


@router.post("/stac/download")
async def stac_download(item_id: str, output_dir: str | None = None) -> dict:
    # Stub -- will be implemented in Phase 2
    return {"status": "not_implemented", "message": "STAC download will be available in Phase 2"}
