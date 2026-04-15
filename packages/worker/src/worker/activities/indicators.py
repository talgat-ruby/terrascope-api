from temporalio import activity


@activity.defn
async def compute_indicators(job_id: str) -> dict:
    """Compute zone-level statistics and save to DB."""
    # TODO Phase 6: Implement using core.services.indicators.IndicatorCalculatorService
    activity.logger.info(f"Computing indicators for job {job_id}")
    return {"status": "computed", "job_id": job_id}
