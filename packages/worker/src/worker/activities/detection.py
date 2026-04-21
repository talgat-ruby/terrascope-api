"""Object detection activities — per-tile parallelism via Temporal."""

import asyncio
import json
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pyproj import Geod
from rasterio.transform import Affine
from sqlalchemy import delete, func, select
from temporalio import activity

from core.config import settings
from core.database import async_session_factory
from core.models.detection import Detection
from core.models.processing import JobStatus
from core.models.tile import Tile
from core.services.detector import DetectorService
from worker.activities._helpers import fail_job, get_job, raw_to_detections, update_job


@dataclass
class TileDetectionInput:
    """Serializable input for a single tile detection activity."""

    job_id: str
    tile_name: str
    tiles_dir: str
    transform: list[float]
    index: list[int]
    pixel_window: list[int]
    crs: str


@activity.defn
async def prepare_detection(job_id: str) -> dict:
    """Read tile manifest, clean old detections, return tile metadata for fan-out."""
    async with async_session_factory() as session:
        job = await get_job(session, job_id)

        # Idempotency guard
        if job.checkpoint_data and "detect" in job.checkpoint_data:
            cached = job.checkpoint_data["detect"]
            return {
                "skipped": True,
                "job_id": job_id,
                "detection_count": cached["raw_detection_count"],
            }

        await update_job(
            session, job, status=JobStatus.DETECTING, current_step="prepare_detection"
        )

        try:
            tile_checkpoint = job.checkpoint_data["tile"]  # type: ignore[index]
            manifest_path = tile_checkpoint["manifest_path"]
            tiles_dir = tile_checkpoint["tiles_dir"]

            manifest = json.loads(Path(manifest_path).read_text())

            # Delete prior detections for idempotency on retry
            await session.execute(
                delete(Detection).where(Detection.job_id == job.id)  # type: ignore[arg-type]
            )
            await session.commit()

            return {
                "skipped": False,
                "job_id": job_id,
                "tiles_dir": tiles_dir,
                "tiles": manifest,
            }

        except Exception as e:
            await fail_job(session, job, str(e))
            raise


@activity.defn
async def detect_tile(input: TileDetectionInput) -> dict:
    """Run ML inference on a single tile and persist detections to DB."""
    tiles_dir = Path(input.tiles_dir)

    data = await asyncio.to_thread(
        np.load, str(tiles_dir / f"{input.tile_name}_data.npy")
    )
    valid_mask = await asyncio.to_thread(
        np.load, str(tiles_dir / f"{input.tile_name}_mask.npy")
    )

    tile = Tile(
        index=(input.index[0], input.index[1]),
        pixel_window=(input.pixel_window[0], input.pixel_window[1], input.pixel_window[2], input.pixel_window[3]),
        transform=Affine(*input.transform),
        data=data,
        valid_mask=valid_mask,
        crs=input.crs,
    )

    detector = DetectorService(device=settings.device)
    await asyncio.to_thread(detector.load_models)

    raw_detections = await asyncio.to_thread(detector.predict_tile, tile)

    geod = Geod(ellps="WGS84")
    db_detections = raw_to_detections(raw_detections, input.job_id, geod)

    async with async_session_factory() as session:
        session.add_all(db_detections)
        await session.commit()

    activity.logger.info(
        f"Tile {input.tile_name}: {len(db_detections)} detections"
    )
    return {"tile_name": input.tile_name, "detection_count": len(db_detections)}


@activity.defn
async def finalize_detection(job_id: str) -> dict:
    """Count persisted detections and update job checkpoint."""
    async with async_session_factory() as session:
        job = await get_job(session, job_id)

        # Idempotency guard
        if job.checkpoint_data and "detect" in job.checkpoint_data:
            cached = job.checkpoint_data["detect"]
            return {
                "status": "detected",
                "job_id": job_id,
                "detection_count": cached["raw_detection_count"],
            }

        result = await session.execute(
            select(func.count()).where(Detection.job_id == uuid.UUID(job_id))  # type: ignore[arg-type]
        )
        total_detections = result.scalar() or 0

        await update_job(
            session,
            job,
            status=JobStatus.DETECTING,
            current_step="detect_objects",
            checkpoint_update={
                "detect": {"raw_detection_count": total_detections}
            },
        )

        activity.logger.info(
            f"Detection finalized for job {job_id}: {total_detections} total detections"
        )
        return {
            "status": "detected",
            "job_id": job_id,
            "detection_count": total_detections,
        }
