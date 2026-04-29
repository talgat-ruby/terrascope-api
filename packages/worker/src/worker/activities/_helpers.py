"""Shared helpers for Temporal activities."""

import uuid
from datetime import UTC, datetime

from geoalchemy2.shape import from_shape, to_shape
from shapely.geometry import Point, box
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select
from temporalio import activity
from temporalio.exceptions import ApplicationError

from core.database import async_session_factory
from core.detection.types import Detection as DomainDetection
from core.models.detection import Detection
from core.models.processing import JobStatus, ProcessingJob


async def get_job(session: AsyncSession, job_id: str) -> ProcessingJob:
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
    job.status = JobStatus.FAILED
    job.error_message = error_message
    job.updated_at = datetime.now(UTC)
    session.add(job)
    await session.commit()


def detections_to_domain(rows: list[Detection]) -> list[DomainDetection]:
    """Hydrate domain Detections from DB rows. `geometry` is now the bbox
    polygon; centroid is recomputed. `pixel_bbox` is unrecoverable after
    persistence so it's set to (0,0,0,0); render_overlay must run alongside
    detection where pixel_bbox is still in scope."""
    out: list[DomainDetection] = []
    for d in rows:
        bbox_poly = to_shape(d.geometry) if d.geometry is not None else None  # type: ignore[arg-type]
        if bbox_poly is None:
            continue
        minx, miny, maxx, maxy = bbox_poly.bounds
        centroid = bbox_poly.centroid
        if not isinstance(centroid, Point):
            continue
        out.append(
            DomainDetection(
                id=d.id,
                class_name=d.class_name,
                confidence=d.confidence,
                bbox=(minx, miny, maxx, maxy),
                pixel_bbox=(0, 0, 0, 0),
                centroid=centroid,
            )
        )
    return out


def domain_to_rows(detections: list[DomainDetection], job_id: str) -> list[Detection]:
    """Convert domain Detections to DB rows with their existing ids."""
    job_uuid = uuid.UUID(job_id)
    rows: list[Detection] = []
    for d in detections:
        bbox_poly = box(*d.bbox)
        rows.append(
            Detection(
                id=d.id,
                job_id=job_uuid,
                class_name=d.class_name,
                confidence=d.confidence,
                geometry=from_shape(bbox_poly, srid=4326),  # type: ignore[arg-type]
            )
        )
    return rows


@activity.defn
async def finalize_job(job_id: str) -> dict:
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
