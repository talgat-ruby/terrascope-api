import logging

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from temporalio.client import Client

from api.dependencies import get_db, get_temporal_client

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health/ready")
async def readiness(
    db: AsyncSession = Depends(get_db),
    temporal: Client = Depends(get_temporal_client),
) -> JSONResponse:
    checks: dict[str, str] = {}

    # Database check
    try:
        await db.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception:
        logger.warning("Health check: database unavailable", exc_info=True)
        checks["database"] = "unavailable"

    # Temporal check
    try:
        await temporal.service_client.check_health()
        checks["temporal"] = "ok"
    except Exception:
        logger.warning("Health check: Temporal unavailable", exc_info=True)
        checks["temporal"] = "unavailable"

    overall = "ok" if all(v == "ok" for v in checks.values()) else "degraded"
    status_code = 200 if overall == "ok" else 503
    return JSONResponse(
        content={"status": overall, "checks": checks},
        status_code=status_code,
    )
