"""Compute indicators activity."""

import asyncio
import uuid

from geoalchemy2.shape import to_shape
from shapely.geometry.base import BaseGeometry
from sqlalchemy import delete
from sqlmodel import select
from temporalio import activity

from core.database import async_session_factory
from core.models.detection import Detection
from core.models.indicator import ZoneIndicator
from core.models.processing import JobStatus
from core.models.territory import Territory
from core.services.indicators import IndicatorCalculatorService
from worker.activities._helpers import (
    detections_to_raw,
    fail_job,
    get_job,
    update_job,
)


@activity.defn
async def compute_indicators(job_id: str) -> dict:
    """Compute zone-level statistics and save to DB."""
    async with async_session_factory() as session:
        job = await get_job(session, job_id)

        # Idempotency guard: skip if already completed
        if job.checkpoint_data and "indicators" in job.checkpoint_data:
            cached = job.checkpoint_data["indicators"]
            return {
                "status": "computed",
                "job_id": job_id,
                "indicator_count": cached["indicator_count"],
            }

        await update_job(
            session,
            job,
            status=JobStatus.COMPUTING_INDICATORS,
            current_step="compute_indicators",
        )

        try:
            # Load detections from DB
            result = await session.execute(
                select(Detection).where(Detection.job_id == job.id)
            )
            db_detections = list(result.scalars().all())
            raw_detections = detections_to_raw(db_detections)

            # Load territory zones
            if job.aoi_id is not None:
                stmt = select(Territory).where(Territory.id == job.aoi_id)
            else:
                stmt = select(Territory)
            result = await session.execute(stmt)
            territories = list(result.scalars().all())

            zones: dict[str, BaseGeometry] = {}
            for territory in territories:
                if territory.geometry is not None:
                    zones[str(territory.id)] = to_shape(territory.geometry)  # type: ignore[arg-type]

            # Compute indicators
            calculator = IndicatorCalculatorService()
            indicators = await asyncio.to_thread(
                calculator.compute, raw_detections, zones
            )

            # Persist to DB
            db_indicators = [
                ZoneIndicator(
                    job_id=job.id,
                    zone_id=uuid.UUID(ind.zone_id),
                    class_name=ind.class_name,
                    count=ind.count,
                    density_per_km2=ind.density_per_km2,
                    total_area_m2=ind.total_area_m2,
                )
                for ind in indicators
            ]
            # Delete any prior indicators for idempotency
            await session.execute(
                delete(ZoneIndicator).where(ZoneIndicator.job_id == job.id)  # type: ignore[arg-type]
            )
            session.add_all(db_indicators)

            await update_job(
                session,
                job,
                status=JobStatus.COMPUTING_INDICATORS,
                current_step="compute_indicators",
                checkpoint_update={
                    "indicators": {
                        "zone_count": len(zones),
                        "indicator_count": len(indicators),
                    }
                },
            )

            activity.logger.info(
                f"Computed {len(indicators)} indicators across "
                f"{len(zones)} zones for job {job_id}"
            )
            return {
                "status": "computed",
                "job_id": job_id,
                "indicator_count": len(indicators),
            }

        except Exception as e:
            await fail_job(session, job, str(e))
            raise
