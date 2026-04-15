import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from api.dependencies import get_db
from core.models.processing import ProcessingJob
from core.schemas.processing import ProcessingRequest, ProcessingStatusResponse

router = APIRouter()


@router.post("/start")
async def start_processing(
    request: ProcessingRequest,
    db: AsyncSession = Depends(get_db),
) -> dict:
    job = ProcessingJob(
        input_path=request.input_path,
        config={
            "aoi": request.aoi,
            "aoi_crs": request.aoi_crs,
            **(request.config or {}),
        },
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    # TODO Phase 7: Start Temporal workflow here
    # client = await get_temporal_client()
    # await client.start_workflow(...)

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
async def retry_processing(job_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> dict:
    result = await db.execute(select(ProcessingJob).where(ProcessingJob.id == job_id))
    job = result.scalar_one_or_none()
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    # TODO Phase 7: Re-trigger Temporal workflow from checkpoint
    return {"job_id": str(job.id), "status": "retry_scheduled"}
