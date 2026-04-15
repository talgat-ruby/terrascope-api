from temporalio import activity


@activity.defn
async def detect_objects(job_id: str) -> dict:
    """Run ML inference on all tiles, save raw detections to DB."""
    # TODO Phase 3: Implement using core.services.detector.DetectorService
    activity.logger.info(f"Detecting objects for job {job_id}")
    return {"status": "detected", "job_id": job_id}
