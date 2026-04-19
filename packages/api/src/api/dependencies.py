from collections.abc import AsyncGenerator

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession
from temporalio.client import Client

from core.config import settings
from core.database import async_session_factory


async def get_db() -> AsyncGenerator[AsyncSession]:
    async with async_session_factory() as session:
        yield session


async def get_temporal_client(request: Request) -> Client:
    client: Client | None = getattr(request.app.state, "temporal_client", None)
    if client is None:
        # Lazy fallback if lifespan connection failed
        client = await Client.connect(settings.temporal_address)
        request.app.state.temporal_client = client
    return client
