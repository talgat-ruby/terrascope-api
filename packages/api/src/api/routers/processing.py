import logging
import uuid
from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select
from temporalio.client import Client

from api.dependencies import get_db, get_temporal_client
from core.config import settings
from core.models.processing import JobStatus, ProcessingJob
from core.schemas.processing import ProcessingRequest, ProcessingStatusResponse
from worker.workflows.processing import ProcessingWorkflow

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/start")
async def start_processing(
    request: ProcessingRequest,
    db: AsyncSession = Depends(get_db),
    temporal: Client = Depends(get_temporal_client),
) -> dict:
    config: dict = {**(request.config or {})}
    if request.aoi is not None:
        config["aoi"] = request.aoi
        config["aoi_crs"] = request.aoi_crs
    job = ProcessingJob(
        input_path=request.input_path,
        config=config,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    try:
        await temporal.start_workflow(
            ProcessingWorkflow.run,
            str(job.id),
            id=f"processing-{job.id}",
            task_queue=settings.temporal_task_queue,
            execution_timeout=timedelta(hours=24),
        )
    except Exception as exc:
        logger.error("Failed to start workflow for job %s: %s", job.id, exc)
        job.status = JobStatus.FAILED
        job.error_message = f"Failed to start workflow: {exc}"
        db.add(job)
        await db.commit()
        raise HTTPException(
            status_code=500, detail="Failed to start processing workflow"
        ) from exc

    return {"job_id": str(job.id), "status": job.status}


@router.get("/{job_id}/status", response_model=ProcessingStatusResponse)
async def get_status(
    job_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> ProcessingJob:
    result = await db.execute(select(ProcessingJob).where(ProcessingJob.id == job_id))
    job = result.scalar_one_or_none()
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.get("/{job_id}/log")
async def get_log(job_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> dict:
    result = await db.execute(select(ProcessingJob).where(ProcessingJob.id == job_id))
    job = result.scalar_one_or_none()
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": str(job.id),
        "current_step": job.current_step,
        "checkpoint_data": job.checkpoint_data,
    }


@router.post("/{job_id}/retry")
async def retry_processing(
    job_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    temporal: Client = Depends(get_temporal_client),
) -> dict:
    result = await db.execute(select(ProcessingJob).where(ProcessingJob.id == job_id))
    job = result.scalar_one_or_none()
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status != JobStatus.FAILED:
        raise HTTPException(
            status_code=400,
            detail=f"Can only retry failed jobs, current status: {job.status}",
        )

    job.status = JobStatus.PENDING
    job.error_message = None
    db.add(job)
    await db.commit()

    try:
        retry_id = uuid.uuid4()
        await temporal.start_workflow(
            ProcessingWorkflow.run,
            str(job.id),
            id=f"processing-{job.id}-retry-{retry_id}",
            task_queue=settings.temporal_task_queue,
            execution_timeout=timedelta(hours=24),
        )
    except Exception as exc:
        logger.error("Failed to start retry workflow for job %s: %s", job.id, exc)
        job.status = JobStatus.FAILED
        job.error_message = f"Failed to start retry workflow: {exc}"
        db.add(job)
        await db.commit()
        raise HTTPException(
            status_code=500, detail="Failed to start retry workflow"
        ) from exc

    return {"job_id": str(job.id), "status": "retry_scheduled"}
