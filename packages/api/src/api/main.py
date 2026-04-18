from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routers import health, imagery, processing, results
from core.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(title="Terrascope API", lifespan=lifespan)

app.include_router(health.router, tags=["health"])
app.include_router(imagery.router, prefix="/imagery", tags=["imagery"])
app.include_router(processing.router, prefix="/processing", tags=["processing"])
app.include_router(results.router, prefix="/results", tags=["results"])

if __name__ == "__main__":
    import uvicorn

    from core.config import settings

    uvicorn.run("api.main:app", host="0.0.0.0", port=settings.api_port, reload=True)
