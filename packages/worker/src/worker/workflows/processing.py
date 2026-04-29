from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from worker.activities._helpers import finalize_job
    from worker.activities.detection import detect
    from worker.activities.export import export_results
    from worker.activities.imagery import load_imagery
    from worker.activities.indicators import compute_indicators

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
    maximum_attempts=3,
    non_retryable_error_types=_non_retryable,
)


@workflow.defn
class ProcessingWorkflow:
    @workflow.run
    async def run(self, job_id: str) -> dict:
        load_result = await workflow.execute_activity(
            load_imagery,
            job_id,
            start_to_close_timeout=timedelta(minutes=10),
            retry_policy=_default_retry,
        )

        detect_result = await workflow.execute_activity(
            detect,
            job_id,
            start_to_close_timeout=timedelta(minutes=30),
            retry_policy=_ml_retry,
        )

        export_result = await workflow.execute_activity(
            export_results,
            job_id,
            start_to_close_timeout=timedelta(minutes=10),
            retry_policy=_default_retry,
        )

        indicator_result = await workflow.execute_activity(
            compute_indicators,
            job_id,
            start_to_close_timeout=timedelta(minutes=10),
            retry_policy=_default_retry,
        )

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
                "detect": detect_result,
                "export": export_result,
                "indicators": indicator_result,
            },
        }
