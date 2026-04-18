"""Post-processing activity."""

import asyncio

from pyproj import Geod
from shapely.geometry import shape as shapely_shape
from sqlalchemy import delete
from sqlmodel import select
from temporalio import activity

from core.config import settings
from core.database import async_session_factory
from core.models.detection import Detection
from core.models.processing import JobStatus
from core.services.postprocessor import PostprocessingConfig, PostprocessorService
from worker.activities._helpers import (
    detections_to_raw,
    fail_job,
    get_job,
    raw_to_detections,
    update_job,
)


@activity.defn
async def postprocess(job_id: str) -> dict:
    """Apply NMS, filtering, and deduplication to detections."""
    async with async_session_factory() as session:
        job = await get_job(session, job_id)
        await update_job(
            session,
            job,
            status=JobStatus.POSTPROCESSING,
            current_step="postprocess",
        )

        try:
            # Load raw detections from DB
            result = await session.execute(
                select(Detection).where(Detection.job_id == job.id)
            )
            db_detections = list(result.scalars().all())
            raw_detections = detections_to_raw(db_detections)

            # Build config from job config with fallbacks
            cfg_overrides = job.config or {}
            config = PostprocessingConfig(
                iou_threshold=cfg_overrides.get(
                    "nms_iou_threshold", settings.nms_iou_threshold
                ),
                confidence_threshold=cfg_overrides.get(
                    "confidence_threshold", settings.confidence_threshold
                ),
                min_area_m2=cfg_overrides.get("min_area_m2", settings.min_area_m2),
                max_area_m2=cfg_overrides.get("max_area_m2", settings.max_area_m2),
                simplify_tolerance_m=cfg_overrides.get(
                    "simplify_tolerance_m", settings.simplify_tolerance_m
                ),
            )

            # Build AOI geometry
            aoi_geom = shapely_shape(cfg_overrides["aoi"])

            # Run postprocessing
            postprocessor = PostprocessorService()
            filtered, stats = await asyncio.to_thread(
                postprocessor.run, raw_detections, config, aoi_geom
            )

            # Delete old detections and insert filtered ones
            await session.execute(delete(Detection).where(Detection.job_id == job.id))
            geod = Geod(ellps="WGS84")
            new_detections = raw_to_detections(filtered, str(job.id), geod)
            session.add_all(new_detections)

            await update_job(
                session,
                job,
                status=JobStatus.POSTPROCESSING,
                current_step="postprocess",
                checkpoint_update={"postprocess": stats},
            )

            activity.logger.info(
                f"Postprocessed job {job_id}: {stats['input_count']} -> "
                f"{stats['output_count']} detections"
            )
            return {"status": "postprocessed", "job_id": job_id, "stats": stats}

        except Exception as e:
            await fail_job(session, job, str(e))
            raise
