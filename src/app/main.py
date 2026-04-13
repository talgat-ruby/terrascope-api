from fastapi import FastAPI

from app.routers import users

app = FastAPI(title="Terrascope API")

app.include_router(users.router, prefix="/users", tags=["users"])


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
