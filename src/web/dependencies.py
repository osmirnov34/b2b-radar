from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.infrastructure.db.session import get_session


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    async with get_session() as session:
        yield session


SessionDep = Annotated[AsyncSession, Depends(get_db_session)]
