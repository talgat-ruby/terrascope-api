from temporalio import activity


@activity.defn
async def postprocess(job_id: str) -> dict:
    """Apply NMS, filtering, and deduplication to detections."""
    # TODO Phase 4: Implement using core.services.postprocessor.PostprocessorService
    activity.logger.info(f"Post-processing detections for job {job_id}")
    return {"status": "postprocessed", "job_id": job_id}
