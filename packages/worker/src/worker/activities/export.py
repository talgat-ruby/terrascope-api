from temporalio import activity


@activity.defn
async def export_results(job_id: str) -> dict:
    """Export detections to GeoJSON, GeoPackage, and Shapefile."""
    # TODO Phase 5: Implement using core.services.exporter.GISExporterService
    activity.logger.info(f"Exporting results for job {job_id}")
    return {"status": "exported", "job_id": job_id}
