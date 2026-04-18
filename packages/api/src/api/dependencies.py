from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession
from temporalio.client import Client

from core.config import settings
from core.database import async_session_factory

_temporal_client: Client | None = None


async def get_db() -> AsyncGenerator[AsyncSession]:
    async with async_session_factory() as session:
        yield session


async def get_temporal_client() -> Client:
    global _temporal_client
    if _temporal_client is None:
        _temporal_client = await Client.connect(settings.temporal_address)
    return _temporal_client
