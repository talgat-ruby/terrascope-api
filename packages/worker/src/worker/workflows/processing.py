import asyncio
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from worker.activities._helpers import finalize_job
    from worker.activities.detection import (
        TileDetectionInput,
        detect_tile,
        finalize_detection,
        prepare_detection,
    )
    from worker.activities.export import export_results
    from worker.activities.imagery import load_imagery, tile_imagery
    from worker.activities.indicators import compute_indicators
    from worker.activities.postprocessing import postprocess

_non_retryable = ["FileNotFoundError", "ValueError"]

_default_retry = RetryPolicy(
    initial_interval=timedelta(seconds=1),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(seconds=60),
    maximum_attempts=3,
    non_retryable_error_types=_non_retryable,
)

_ml_retry = RetryPolicy(
    initial_interval=timedelta(seconds=2),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(seconds=120),
    maximum_attempts=5,
    non_retryable_error_types=_non_retryable,
)


@workflow.defn
class ProcessingWorkflow:
    @workflow.run
    async def run(self, job_id: str) -> dict:
        # Step 1: Load and validate imagery
        load_result = await workflow.execute_activity(
            load_imagery,
            job_id,
            start_to_close_timeout=timedelta(minutes=10),
            retry_policy=_default_retry,
        )

        # Step 2: Tile the imagery
        tile_result = await workflow.execute_activity(
            tile_imagery,
            job_id,
            start_to_close_timeout=timedelta(minutes=10),
            retry_policy=_default_retry,
        )

        # Step 3a: Prepare detection (read manifest, clean old detections)
        prep_result = await workflow.execute_activity(
            prepare_detection,
            job_id,
            start_to_close_timeout=timedelta(minutes=5),
            retry_policy=_default_retry,
        )

        # Step 3b: Fan-out per-tile detection across available workers
        if prep_result["skipped"]:
            detection_result = prep_result
        else:
            tile_futures = [
                workflow.execute_activity(
                    detect_tile,
                    TileDetectionInput(
                        job_id=job_id,
                        tile_name=entry["name"],
                        tiles_dir=prep_result["tiles_dir"],
                        transform=entry["transform"],
                        index=entry["index"],
                        pixel_window=entry["pixel_window"],
                        crs=entry["crs"],
                    ),
                    start_to_close_timeout=timedelta(minutes=10),
                    retry_policy=_ml_retry,
                )
                for entry in prep_result["tiles"]
            ]
            await asyncio.gather(*tile_futures)

            # Step 3c: Finalize detection (count + checkpoint)
            detection_result = await workflow.execute_activity(
                finalize_detection,
                job_id,
                start_to_close_timeout=timedelta(minutes=2),
                retry_policy=_default_retry,
            )

        # Step 4: Post-process detections
        postprocess_result = await workflow.execute_activity(
            postprocess,
            job_id,
            start_to_close_timeout=timedelta(minutes=10),
            retry_policy=_default_retry,
        )

        # Step 5: Export results to GIS formats
        export_result = await workflow.execute_activity(
            export_results,
            job_id,
            start_to_close_timeout=timedelta(minutes=10),
            retry_policy=_default_retry,
        )

        # Step 6: Compute zone indicators
        indicator_result = await workflow.execute_activity(
            compute_indicators,
            job_id,
            start_to_close_timeout=timedelta(minutes=10),
            retry_policy=_default_retry,
        )

        # Step 7: Mark job as completed
        await workflow.execute_activity(
            finalize_job,
            job_id,
            start_to_close_timeout=timedelta(minutes=1),
            retry_policy=_default_retry,
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
