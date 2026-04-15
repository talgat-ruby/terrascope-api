import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from api.dependencies import get_db
from core.models.detection import Detection
from core.models.indicator import ZoneIndicator
from core.models.processing import ProcessingJob
from core.models.quality import QualityMetrics

router = APIRouter()


@router.get("/{job_id}/detections")
async def get_detections(
    job_id: uuid.UUID,
    class_name: str | None = Query(None, alias="class"),
    confidence_min: int | None = Query(None, ge=0, le=100),
    db: AsyncSession = Depends(get_db),
) -> dict:
    stmt = select(Detection).where(Detection.job_id == job_id)
    if class_name:
        stmt = stmt.where(Detection.class_name == class_name)
    if confidence_min is not None:
        stmt = stmt.where(Detection.confidence >= confidence_min)

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
                "geometry": None,  # TODO: convert from WKB via ST_AsGeoJSON
            }
            for d in detections
        ],
    }


@router.get("/{job_id}/download")
async def download_results(
    job_id: uuid.UUID,
    format: str = Query("geojson", pattern="^(geojson|gpkg|shp)$"),
) -> dict:
    # TODO Phase 5: Return FileResponse with exported file
    return {"status": "not_implemented", "message": "Export will be available in Phase 5"}


@router.get("/{job_id}/indicators")
async def get_indicators(
    job_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> list[dict]:
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
