"""Shared helpers for Temporal activities."""

import uuid
from datetime import UTC, datetime

from geoalchemy2.shape import from_shape, to_shape
from pyproj import Geod
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select
from temporalio import activity
from temporalio.exceptions import ApplicationError

from core.database import async_session_factory
from core.models.detection import Detection
from core.models.processing import JobStatus, ProcessingJob
from core.services.detector import RawDetection


async def get_job(session: AsyncSession, job_id: str) -> ProcessingJob:
    """Fetch a ProcessingJob by ID or raise a non-retryable error."""
    result = await session.execute(
        select(ProcessingJob).where(ProcessingJob.id == uuid.UUID(job_id))
    )
    job = result.scalar_one_or_none()
    if job is None:
        raise ApplicationError(f"Job {job_id} not found", non_retryable=True)
    return job


async def update_job(
    session: AsyncSession,
    job: ProcessingJob,
    *,
    status: JobStatus,
    current_step: str,
    checkpoint_update: dict | None = None,
    completed: bool = False,
) -> None:
    """Update job status, current_step, and checkpoint_data, then commit."""
    job.status = status
    job.current_step = current_step
    job.updated_at = datetime.now(UTC)

    if checkpoint_update is not None:
        if job.checkpoint_data is None:
            job.checkpoint_data = {}
        job.checkpoint_data = {**job.checkpoint_data, **checkpoint_update}

    if completed:
        job.completed_at = datetime.now(UTC)

    session.add(job)
    await session.commit()


async def fail_job(
    session: AsyncSession, job: ProcessingJob, error_message: str
) -> None:
    """Mark a job as failed with an error message."""
    job.status = JobStatus.FAILED
    job.error_message = error_message
    job.updated_at = datetime.now(UTC)
    session.add(job)
    await session.commit()


def detections_to_raw(detections: list[Detection]) -> list[RawDetection]:
    """Convert DB Detection rows to RawDetection list."""
    return [
        RawDetection(
            class_name=d.class_name,
            confidence=d.confidence,
            geometry=to_shape(d.geometry),  # type: ignore[arg-type]
            source=d.source,
        )
        for d in detections
    ]


def raw_to_detections(
    raw_detections: list[RawDetection],
    job_id: str,
    geod: Geod,
) -> list[Detection]:
    """Convert RawDetection list to DB Detection models."""
    results: list[Detection] = []
    for raw in raw_detections:
        area_m2 = abs(geod.geometry_area_perimeter(raw.geometry)[0])
        results.append(
            Detection(
                job_id=uuid.UUID(job_id),
                class_name=raw.class_name,
                confidence=raw.confidence,
                source=raw.source,
                geometry=from_shape(raw.geometry, srid=4326),  # type: ignore[arg-type]
                area_m2=round(area_m2, 2),
            )
        )
    return results


@activity.defn
async def finalize_job(job_id: str) -> dict:
    """Mark job as completed after all activities succeed."""
    async with async_session_factory() as session:
        job = await get_job(session, job_id)
        await update_job(
            session,
            job,
            status=JobStatus.COMPLETED,
            current_step="completed",
            completed=True,
        )
        return {"status": "completed", "job_id": job_id}
