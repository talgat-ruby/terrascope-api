"""Object detection activity."""

import asyncio
import json
from pathlib import Path

import numpy as np
from pyproj import Geod
from rasterio.transform import Affine
from sqlalchemy import delete
from temporalio import activity

from core.config import settings
from core.database import async_session_factory
from core.models.detection import Detection
from core.models.processing import JobStatus
from core.models.tile import Tile
from core.services.detector import DetectorService
from worker.activities._helpers import fail_job, get_job, raw_to_detections, update_job


@activity.defn
async def detect_objects(job_id: str) -> dict:
    """Run ML inference on all tiles, save raw detections to DB."""
    async with async_session_factory() as session:
        job = await get_job(session, job_id)

        # Idempotency guard: skip if already completed
        if job.checkpoint_data and "detect" in job.checkpoint_data:
            cached = job.checkpoint_data["detect"]
            return {
                "status": "detected",
                "job_id": job_id,
                "detection_count": cached["raw_detection_count"],
            }

        await update_job(
            session, job, status=JobStatus.DETECTING, current_step="detect_objects"
        )

        try:
            tile_checkpoint = job.checkpoint_data["tile"]  # type: ignore[index]
            manifest_path = Path(tile_checkpoint["manifest_path"])
            tiles_dir = Path(tile_checkpoint["tiles_dir"])

            # Load tiles from manifest
            manifest = json.loads(manifest_path.read_text())
            tiles: list[Tile] = []
            for entry in manifest:
                tile_name = entry["name"]
                data = await asyncio.to_thread(
                    np.load, str(tiles_dir / f"{tile_name}_data.npy")
                )
                valid_mask = await asyncio.to_thread(
                    np.load, str(tiles_dir / f"{tile_name}_mask.npy")
                )
                transform = Affine(*entry["transform"])
                tiles.append(
                    Tile(
                        index=tuple(entry["index"]),
                        pixel_window=tuple(entry["pixel_window"]),
                        transform=transform,
                        data=data,
                        valid_mask=valid_mask,
                        crs=entry["crs"],
                    )
                )

            # Run detection
            detector = DetectorService(device=settings.device)
            await asyncio.to_thread(detector.load_models)

            all_raw_detections = []
            for i, tile in enumerate(tiles):
                tile_detections = await asyncio.to_thread(detector.predict_tile, tile)
                all_raw_detections.extend(tile_detections)
                activity.logger.info(
                    f"Tile {i + 1}/{len(tiles)}: {len(tile_detections)} detections"
                )

            # Persist to DB (delete any prior detections for idempotency)
            await session.execute(
                delete(Detection).where(Detection.job_id == job.id)  # type: ignore[arg-type]
            )
            geod = Geod(ellps="WGS84")
            db_detections = raw_to_detections(all_raw_detections, job_id, geod)
            session.add_all(db_detections)

            await update_job(
                session,
                job,
                status=JobStatus.DETECTING,
                current_step="detect_objects",
                checkpoint_update={
                    "detect": {"raw_detection_count": len(db_detections)}
                },
            )

            activity.logger.info(
                f"Detected {len(db_detections)} objects for job {job_id}"
            )
            return {
                "status": "detected",
                "job_id": job_id,
                "detection_count": len(db_detections),
            }

        except Exception as e:
            await fail_job(session, job, str(e))
            raise
