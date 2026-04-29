"""Export activity: write GeoJSON from persisted detections."""

import asyncio

from sqlmodel import select
from temporalio import activity

from core.config import settings
from core.database import async_session_factory
from core.models.detection import Detection
from core.models.processing import JobStatus
from core.services.exporter import GISExporterService
from worker.activities._helpers import (
    detections_to_domain,
    fail_job,
    get_job,
    update_job,
)


@activity.defn
async def export_results(job_id: str) -> dict:
    """Write GeoJSON for the job's detections. PNG was already rendered
    during the detection activity (where pixel_bbox is still in scope)."""
    async with async_session_factory() as session:
        job = await get_job(session, job_id)
        await update_job(
            session, job, status=JobStatus.EXPORTING, current_step="export_results"
        )

        try:
            result = await session.execute(
                select(Detection).where(Detection.job_id == job.id)
            )
            rows = list(result.scalars().all())
            detections = detections_to_domain(rows)

            crs = job.checkpoint_data["load"]["crs"]  # type: ignore[index]
            overlay_path = job.checkpoint_data.get("detect", {}).get(  # type: ignore[union-attr]
                "overlay_path"
            )

            output_dir = settings.output_dir / job_id
            output_dir.mkdir(parents=True, exist_ok=True)
            geojson_path = output_dir / "detections.geojson"

            exporter = GISExporterService()
            await asyncio.to_thread(
                exporter.export_geojson, detections, geojson_path, crs
            )

            formats = {"geojson": str(geojson_path)}
            if overlay_path:
                formats["png"] = overlay_path

            await update_job(
                session,
                job,
                status=JobStatus.EXPORTING,
                current_step="export_results",
                checkpoint_update={"export": {"formats": formats}},
            )

            activity.logger.info(
                f"Exported {len(detections)} detections for job {job_id}"
            )
            return {
                "status": "exported",
                "job_id": job_id,
                "formats": list(formats.keys()),
            }

        except Exception as e:
            await fail_job(session, job, str(e))
            raise
