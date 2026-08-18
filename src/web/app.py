import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from src.config import get_ml_settings
from src.infrastructure.db.repositories import SourceRepository
from src.infrastructure.db.session import get_session
from src.web.ml_snapshot import MLSnapshotRepository
from src.web.routers import analysis, api_keys, dashboard, documents, ml_topics, sources

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parents[2] / "web" / "static"

# A single source is RUNNING for only the minutes it takes to fetch one video's comments, so
# anything RUNNING far longer is a crashed/orphaned ingest. Well above that window yet well below
# "stuck forever", and deliberately longer than any live source so a concurrent CLI backfill during
# a web restart is never falsely failed. See SourceRepository.reset_stale_running.
_STALE_INGEST_THRESHOLD = timedelta(minutes=30)


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    # Reap ingests orphaned by a crash/restart (BackgroundTasks can't survive the process): flip
    # long-stalled RUNNING sources to FAILED so they're visible and retryable instead of stuck
    # forever with no error. Age-scoped so a live, concurrent ingest is never touched.
    async with get_session() as session:
        reset = await SourceRepository(session).reset_stale_running(_STALE_INGEST_THRESHOLD)
        await session.commit()
    if reset:
        logger.warning("Reset %d stale RUNNING source(s) to FAILED on startup.", reset)
    yield


def create_app(*, ml_repository: MLSnapshotRepository | None = None) -> FastAPI:
    app = FastAPI(title="B2B Radar", lifespan=_lifespan)

    if ml_repository is None:
        ml_settings = get_ml_settings()
        ml_repository = MLSnapshotRepository(
            ml_settings.ml_export_manifest,
            enabled=ml_settings.ml_results_enabled,
            allow_unreliable=ml_settings.ml_allow_unreliable,
        )
    app.state.ml_repository = ml_repository

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    app.include_router(dashboard.router)
    app.include_router(sources.router)
    app.include_router(documents.router)
    app.include_router(analysis.router)
    app.include_router(api_keys.router)
    app.include_router(ml_topics.router)

    return app


app = create_app()
