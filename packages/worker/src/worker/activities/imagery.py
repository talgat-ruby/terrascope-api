from temporalio import activity


@activity.defn
async def load_imagery(job_id: str) -> dict:
    """Load and validate GeoTIFF imagery, save metadata to DB."""
    # TODO Phase 2: Implement using core.services.imagery.ImageryLoaderService
    activity.logger.info(f"Loading imagery for job {job_id}")
    return {"status": "loaded", "job_id": job_id}


@activity.defn
async def tile_imagery(job_id: str) -> dict:
    """Generate tiles from imagery, store tile manifest as checkpoint."""
    # TODO Phase 2: Implement using core.services.tiler.TilerService
    activity.logger.info(f"Tiling imagery for job {job_id}")
    return {"status": "tiled", "job_id": job_id}
