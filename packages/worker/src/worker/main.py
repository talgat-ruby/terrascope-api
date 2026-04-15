import asyncio

from temporalio.client import Client
from temporalio.worker import Worker

from core.config import settings
from worker.activities.detection import detect_objects
from worker.activities.export import export_results
from worker.activities.imagery import load_imagery, tile_imagery
from worker.activities.indicators import compute_indicators
from worker.activities.postprocessing import postprocess
from worker.workflows.processing import ProcessingWorkflow


async def run_worker() -> None:
    client = await Client.connect(settings.temporal_host)

    worker = Worker(
        client,
        task_queue=settings.temporal_task_queue,
        workflows=[ProcessingWorkflow],
        activities=[
            load_imagery,
            tile_imagery,
            detect_objects,
            postprocess,
            export_results,
            compute_indicators,
        ],
    )

    print(f"Worker started, listening on queue: {settings.temporal_task_queue}")
    await worker.run()


def main() -> None:
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
