import os

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.domain.source import Source, SourceType
from src.infrastructure.db.models import Base
from src.infrastructure.db.repositories import SourceRepository


@pytest.mark.integration
@pytest.mark.asyncio
async def test_source_repository_round_trip() -> None:
    database_url = os.getenv("TEST_DATABASE_URL")
    if database_url is None:
        pytest.skip("TEST_DATABASE_URL is not configured")

    engine = create_async_engine(database_url)
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.drop_all)
            await connection.run_sync(Base.metadata.create_all)

        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        async with session_factory() as session:
            repository = SourceRepository(session)
            saved = await repository.add(
                Source(
                    type=SourceType.YOUTUBE_VIDEO,
                    url="https://www.youtube.com/watch?v=integration-test",
                    name="Integration test",
                ),
            )
            await session.flush()
            loaded = await repository.get_by_url(saved.url)

        assert saved.id is not None
        assert loaded is not None
        assert loaded.id == saved.id
    finally:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.drop_all)
        await engine.dispose()
