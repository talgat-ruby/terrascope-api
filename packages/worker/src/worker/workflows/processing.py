from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from worker.activities.detection import detect_objects
    from worker.activities.export import export_results
    from worker.activities.imagery import load_imagery, tile_imagery
    from worker.activities.indicators import compute_indicators
    from worker.activities.postprocessing import postprocess


@workflow.defn
class ProcessingWorkflow:
    @workflow.run
    async def run(self, job_id: str) -> dict:
        # Step 1: Load and validate imagery
        load_result = await workflow.execute_activity(
            load_imagery,
            job_id,
            start_to_close_timeout=timedelta(minutes=10),
        )

        # Step 2: Tile the imagery
        tile_result = await workflow.execute_activity(
            tile_imagery,
            job_id,
            start_to_close_timeout=timedelta(minutes=10),
        )

        # Step 3: Run ML detection on tiles
        detection_result = await workflow.execute_activity(
            detect_objects,
            job_id,
            start_to_close_timeout=timedelta(minutes=60),
        )

        # Step 4: Post-process detections
        postprocess_result = await workflow.execute_activity(
            postprocess,
            job_id,
            start_to_close_timeout=timedelta(minutes=10),
        )

        # Step 5: Export results to GIS formats
        export_result = await workflow.execute_activity(
            export_results,
            job_id,
            start_to_close_timeout=timedelta(minutes=10),
        )

        # Step 6: Compute zone indicators
        indicator_result = await workflow.execute_activity(
            compute_indicators,
            job_id,
            start_to_close_timeout=timedelta(minutes=10),
        )

        return {
            "job_id": job_id,
            "status": "completed",
            "steps": {
                "load": load_result,
                "tile": tile_result,
                "detect": detection_result,
                "postprocess": postprocess_result,
                "export": export_result,
                "indicators": indicator_result,
            },
        }
