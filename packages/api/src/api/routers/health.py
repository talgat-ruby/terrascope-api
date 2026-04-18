from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import get_db, get_temporal_client

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health/ready")
async def readiness(db: AsyncSession = Depends(get_db)) -> JSONResponse:
    checks: dict[str, str] = {}

    # Database check
    try:
        await db.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as e:
        checks["database"] = f"error: {e}"

    # Temporal check
    try:
        client = await get_temporal_client()
        await client.service_client.check_health()
        checks["temporal"] = "ok"
    except Exception as e:
        checks["temporal"] = f"error: {e}"

    overall = "ok" if all(v == "ok" for v in checks.values()) else "degraded"
    status_code = 200 if overall == "ok" else 503
    return JSONResponse(
        content={"status": overall, "checks": checks},
        status_code=status_code,
    )
