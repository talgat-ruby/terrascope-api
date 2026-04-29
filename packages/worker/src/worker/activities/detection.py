"""Detection activity: load raster, run detector, filter, persist."""

import asyncio
import uuid

import numpy as np
from rasterio.transform import Affine
from shapely import wkt as shapely_wkt
from sqlalchemy import delete
from temporalio import activity

from core.config import settings
from core.database import async_session_factory
from core.detection import build_detector, filter_detections, render_overlay
from core.detection.types import Raster
from core.models.detection import Detection
from core.models.processing import JobStatus
from worker.activities._helpers import (
    domain_to_rows,
    fail_job,
    get_job,
    update_job,
)


@activity.defn
async def detect(job_id: str) -> dict:
    """Run the configured Detector against the loaded raster, persist results."""
    async with async_session_factory() as session:
        job = await get_job(session, job_id)
        await update_job(
            session, job, status=JobStatus.DETECTING, current_step="detect"
        )

        try:
            load_checkpoint = job.checkpoint_data["load"]  # type: ignore[index]
            data = await asyncio.to_thread(np.load, load_checkpoint["clipped_path"])
            transform = Affine(*load_checkpoint["transform"])
            crs = load_checkpoint["crs"]
            aoi_wkt = load_checkpoint.get("aoi_wkt")
            aoi_geom = shapely_wkt.loads(aoi_wkt) if aoi_wkt else None

            raster = Raster(data=data, transform=transform, crs=crs, aoi_geom=aoi_geom)

            detector_name = (job.config or {}).get("detector_name")
            detector = await asyncio.to_thread(build_detector, detector_name)

            min_confidence = (job.config or {}).get("min_confidence", 0.25)

            detections = await asyncio.to_thread(detector.detect, raster)
            detections = filter_detections(
                detections, min_confidence=min_confidence, aoi=aoi_geom
            )

            # Persist with the sequential ids assigned by filter_detections.
            await session.execute(
                delete(Detection).where(Detection.job_id == uuid.UUID(job_id))  # type: ignore[arg-type]
            )
            rows = domain_to_rows(detections, job_id)
            session.add_all(rows)

            # Render PNG overlay while pixel_bbox is still in scope (it's
            # not persisted to the DB). The export activity that follows
            # only needs geographic bboxes from the DB rows.
            job_dir = settings.output_dir / job_id
            job_dir.mkdir(parents=True, exist_ok=True)
            overlay_path = job_dir / "overlay.png"
            await asyncio.to_thread(render_overlay, raster, detections, overlay_path)

            await update_job(
                session,
                job,
                status=JobStatus.DETECTING,
                current_step="detect",
                checkpoint_update={
                    "detect": {
                        "detector": detector.name,
                        "detection_count": len(detections),
                        "overlay_path": str(overlay_path),
                    }
                },
            )

            activity.logger.info(
                f"Detection done for job {job_id}: {len(detections)} via {detector.name}"
            )
            return {
                "status": "detected",
                "job_id": job_id,
                "detection_count": len(detections),
            }

        except Exception as e:
            await fail_job(session, job, str(e))
            raise
