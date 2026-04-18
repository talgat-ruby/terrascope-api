"""Export results activity."""

import asyncio

from sqlmodel import select
from temporalio import activity

from core.config import settings
from core.database import async_session_factory
from core.models.detection import Detection
from core.models.processing import JobStatus
from core.services.exporter import GISExporterService
from worker.activities._helpers import (
    detections_to_raw,
    fail_job,
    get_job,
    update_job,
)


@activity.defn
async def export_results(job_id: str) -> dict:
    """Export detections to GeoJSON, GeoPackage, and Shapefile."""
    async with async_session_factory() as session:
        job = await get_job(session, job_id)
        await update_job(
            session, job, status=JobStatus.EXPORTING, current_step="export_results"
        )

        try:
            # Load detections from DB
            result = await session.execute(
                select(Detection).where(Detection.job_id == job.id)
            )
            db_detections = list(result.scalars().all())
            raw_detections = detections_to_raw(db_detections)

            # Get CRS from load checkpoint
            crs = job.checkpoint_data["load"]["crs"]  # type: ignore[index]

            # Export to all formats
            output_dir = settings.output_dir / job_id / "exports"
            exporter = GISExporterService()
            paths = await asyncio.to_thread(
                exporter.export_all, raw_detections, crs, output_dir
            )

            format_paths = {k: str(v) for k, v in paths.items()}

            await update_job(
                session,
                job,
                status=JobStatus.EXPORTING,
                current_step="export_results",
                checkpoint_update={"export": {"formats": format_paths}},
            )

            activity.logger.info(
                f"Exported {len(raw_detections)} detections for job {job_id}: "
                f"{list(format_paths.keys())}"
            )
            return {
                "status": "exported",
                "job_id": job_id,
                "formats": list(format_paths.keys()),
            }

        except Exception as e:
            await fail_job(session, job, str(e))
            raise
