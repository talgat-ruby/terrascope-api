import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from temporalio.client import Client

from api.routers import health, imagery, processing, results
from core.config import settings
from core.database import init_db

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    try:
        app.state.temporal_client = await Client.connect(settings.temporal_address)
        logger.info("Connected to Temporal at %s", settings.temporal_address)
    except Exception:
        logger.warning("Could not connect to Temporal at startup", exc_info=True)
        app.state.temporal_client = None
    yield


app = FastAPI(title="Terrascope API", lifespan=lifespan)

app.include_router(health.router, tags=["health"])
app.include_router(imagery.router, prefix="/imagery", tags=["imagery"])
app.include_router(processing.router, prefix="/processing", tags=["processing"])
app.include_router(results.router, prefix="/results", tags=["results"])

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app", host="0.0.0.0", port=settings.api_port, reload=settings.debug
    )
