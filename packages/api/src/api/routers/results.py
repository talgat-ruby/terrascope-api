import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from geoalchemy2.shape import to_shape
from shapely.geometry import mapping
from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from api.dependencies import get_db
from core.models.detection import Detection
from core.models.indicator import ZoneIndicator
from core.models.processing import JobStatus, ProcessingJob
from core.models.quality import QualityMetrics

router = APIRouter()


async def _check_job_exists(job_id: uuid.UUID, db: AsyncSession) -> None:
    result = await db.execute(select(ProcessingJob).where(ProcessingJob.id == job_id))
    if result.scalar_one_or_none() is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")


@router.get("/{job_id}/detections")
async def get_detections(
    job_id: uuid.UUID,
    class_name: str | None = Query(None, alias="class"),
    confidence_min: int | None = Query(None, ge=0, le=100),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> dict:
    await _check_job_exists(job_id, db)

    stmt = select(Detection).where(Detection.job_id == job_id)
    if class_name:
        stmt = stmt.where(Detection.class_name == class_name)
    if confidence_min is not None:
        stmt = stmt.where(Detection.confidence >= confidence_min)

    count_stmt = select(func.count()).select_from(stmt.subquery())
    total = (await db.execute(count_stmt)).scalar_one()

    stmt = stmt.offset(offset).limit(limit)
    result = await db.execute(stmt)
    detections = result.scalars().all()

    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "id": str(d.id),
                    "class": d.class_name,
                    "confidence": d.confidence,
                    "source": d.source,
                    "area_m2": d.area_m2,
                    "length_m": d.length_m,
                },
                "geometry": mapping(to_shape(d.geometry)) if d.geometry else None,
            }
            for d in detections
        ],
        "pagination": {
            "total": total,
            "limit": limit,
            "offset": offset,
        },
    }


@router.get("/{job_id}/download")
async def download_results(
    job_id: uuid.UUID,
    format: str = Query("geojson", pattern="^(geojson|gpkg|shp)$"),
    db: AsyncSession = Depends(get_db),
) -> FileResponse:
    result = await db.execute(select(ProcessingJob).where(ProcessingJob.id == job_id))
    job = result.scalar_one_or_none()
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job not completed, current status: {job.status}",
        )

    export_formats = (job.checkpoint_data or {}).get("export", {}).get("formats", {})
    file_path_str = export_formats.get(format)
    if not file_path_str:
        raise HTTPException(
            status_code=404,
            detail=f"Export format '{format}' not available",
        )

    file_path = Path(file_path_str)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Export file not found on disk")

    media_types = {
        "geojson": "application/geo+json",
        "gpkg": "application/geopackage+sqlite3",
        "shp": "application/x-shapefile",
    }
    return FileResponse(
        path=str(file_path),
        filename=file_path.name,
        media_type=media_types.get(format, "application/octet-stream"),
    )


@router.get("/{job_id}/indicators")
async def get_indicators(
    job_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> list[dict]:
    await _check_job_exists(job_id, db)

    result = await db.execute(
        select(ZoneIndicator).where(ZoneIndicator.job_id == job_id)
    )
    indicators = result.scalars().all()
    return [
        {
            "zone_id": str(i.zone_id),
            "class_name": i.class_name,
            "count": i.count,
            "density_per_km2": i.density_per_km2,
            "total_area_m2": i.total_area_m2,
        }
        for i in indicators
    ]


@router.get("/{job_id}/quality")
async def get_quality(
    job_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> dict:
    await _check_job_exists(job_id, db)

    result = await db.execute(
        select(QualityMetrics).where(QualityMetrics.job_id == job_id)
    )
    metrics = result.scalars().all()
    return {
        "job_id": str(job_id),
        "metrics": [
            {
                "class_name": m.class_name,
                "precision": m.precision,
                "recall": m.recall,
                "f1": m.f1,
                "iou": m.iou,
                "map": m.map,
            }
            for m in metrics
        ],
    }
