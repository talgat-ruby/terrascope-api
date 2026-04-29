"""Imagery loading activity."""

import asyncio

import numpy as np
from shapely.geometry import shape as shapely_shape
from temporalio import activity

from core.config import settings
from core.database import async_session_factory
from core.models.processing import JobStatus
from core.services.imagery import ImageryLoaderService
from worker.activities._helpers import fail_job, get_job, update_job


@activity.defn
async def load_imagery(job_id: str) -> dict:
    """Load + clip the GeoTIFF, persist as a uint8 RGB Raster blob on disk."""
    async with async_session_factory() as session:
        job = await get_job(session, job_id)
        await update_job(
            session, job, status=JobStatus.LOADING, current_step="load_imagery"
        )

        try:
            loader = ImageryLoaderService()

            aoi_geojson = (job.config or {}).get("aoi")
            aoi_geom = shapely_shape(aoi_geojson) if aoi_geojson is not None else None
            aoi_crs = (job.config or {}).get("aoi_crs", "EPSG:4326")

            raster = await asyncio.to_thread(
                loader.load_clipped, job.input_path, aoi_geom, aoi_crs
            )

            job_dir = settings.output_dir / job_id
            job_dir.mkdir(parents=True, exist_ok=True)
            clipped_path = job_dir / "clipped.npy"
            await asyncio.to_thread(np.save, str(clipped_path), raster.data)

            transform = raster.transform
            transform_list = [
                transform.a,
                transform.b,
                transform.c,
                transform.d,
                transform.e,
                transform.f,
            ]
            aoi_wkt = raster.aoi_geom.wkt if raster.aoi_geom is not None else None

            await update_job(
                session,
                job,
                status=JobStatus.LOADING,
                current_step="load_imagery",
                checkpoint_update={
                    "load": {
                        "crs": raster.crs,
                        "clipped_path": str(clipped_path),
                        "transform": transform_list,
                        "shape": list(raster.data.shape),
                        "aoi_wkt": aoi_wkt,
                    }
                },
            )

            activity.logger.info(
                f"Loaded imagery for job {job_id}: shape={raster.data.shape}, CRS={raster.crs}"
            )
            return {"status": "loaded", "job_id": job_id, "crs": raster.crs}

        except Exception as e:
            await fail_job(session, job, str(e))
            raise
