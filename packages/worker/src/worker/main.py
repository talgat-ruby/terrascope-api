import asyncio
import logging
import signal

from temporalio.client import Client
from temporalio.worker import Worker

from core.config import settings
from worker.activities._helpers import finalize_job
from worker.activities.detection import detect
from worker.activities.export import export_results
from worker.activities.imagery import load_imagery
from worker.activities.indicators import compute_indicators
from worker.workflows.processing import ProcessingWorkflow

logger = logging.getLogger(__name__)


async def run_worker() -> None:
    client = await Client.connect(settings.temporal_address)

    worker = Worker(
        client,
        task_queue=settings.temporal_task_queue,
        workflows=[ProcessingWorkflow],
        activities=[
            load_imagery,
            detect,
            export_results,
            compute_indicators,
            finalize_job,
        ],
    )

    shutdown_event = asyncio.Event()

    def _signal_handler() -> None:
        logger.info("Shutdown signal received, draining activities...")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _signal_handler)

    logger.info("Worker started, listening on queue: %s", settings.temporal_task_queue)

    async with worker:
        await shutdown_event.wait()

    logger.info("Worker shut down gracefully")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
