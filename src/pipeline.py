import asyncio
import logging
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from src.domain.source import IngestStatus, Source
from src.infrastructure.api.youtube import YoutubeClient
from src.infrastructure.db.repositories import (
    DocumentRepository,
    FilterSettingsRepository,
    SourceRepository,
    YoutubeApiKeyRepository,
)
from src.infrastructure.db.session import get_session
from src.infrastructure.extractor.quality import YoutubeQualityFilter
from src.infrastructure.extractor.youtube import YoutubeExtractor

logger = logging.getLogger(__name__)


@dataclass
class _IngestContext:
    """Bundles the collaborators a single pipeline run shares, to keep helper signatures small."""

    session: AsyncSession
    extractor: YoutubeExtractor
    quality_filter: YoutubeQualityFilter
    source_repo: SourceRepository
    document_repo: DocumentRepository


async def _save_sources(ctx: _IngestContext, query: str, limit: int) -> list[Source]:
    # extract_sources performs blocking network I/O; run it off the event loop.
    extracted = await asyncio.to_thread(ctx.extractor.extract_sources, query, limit)
    sources = [source for source in extracted if ctx.quality_filter.accepts_video(source)]
    logger.info("Video quality filter kept %d/%d source(s)", len(sources), len(extracted))
    saved_sources = []

    for source in sources:
        existing = await ctx.source_repo.get_by_url(source.url)
        if existing is not None:
            saved_sources.append(existing)
            continue
        source.metadata["search_query"] = query
        saved_sources.append(await ctx.source_repo.add(source))

    return saved_sources


async def _extract_documents_for_source(ctx: _IngestContext, source: Source, limit: int) -> int:
    # extract_documents performs blocking network I/O; run it off the event loop.
    extracted = await asyncio.to_thread(ctx.extractor.extract_documents, source, limit)
    documents = [document for document in extracted if ctx.quality_filter.accepts_comment(document)]
    if not documents:
        return 0

    existing_ids = await ctx.document_repo.list_existing_external_ids([document.external_id for document in documents])
    new_documents = [document for document in documents if document.external_id not in existing_ids]

    if new_documents:
        await ctx.document_repo.add_many(new_documents)
    return len(new_documents)


async def _save_documents(ctx: _IngestContext, sources: list[Source], limit: int) -> int:
    """Extract documents for each source, isolating failures so one bad video can't abort the run.

    Every source's outcome (running/success/failed) is committed as it happens, so the UI reflects
    live progress and any failure is visible instead of the source hanging in ``pending`` forever.
    """
    saved_count = 0

    for source in sources:
        if source.id is None:
            continue

        await ctx.source_repo.set_ingest_status(source.id, IngestStatus.RUNNING)
        await ctx.session.commit()

        try:
            saved_count += await _extract_documents_for_source(ctx, source, limit)
        except Exception as exc:  # one source's failure must not abort the whole run
            logger.exception("Document extraction failed for source_id=%d", source.id)
            await ctx.session.rollback()
            await ctx.source_repo.set_ingest_status(source.id, IngestStatus.FAILED, str(exc))
        else:
            await ctx.source_repo.set_ingest_status(source.id, IngestStatus.SUCCESS)
        await ctx.session.commit()

    return saved_count


async def _build_extractor() -> YoutubeExtractor:
    async with get_session() as session:
        api_keys = await YoutubeApiKeyRepository(session).list_active_keys()
    if not api_keys:
        msg = "No active YouTube API keys found in the youtube_api_keys table."
        raise RuntimeError(msg)
    return YoutubeExtractor(YoutubeClient(api_keys))


async def run(query: str, source_limit: int, document_limit: int) -> None:
    """Run the full pipeline: discover sources for a query, then extract their documents."""
    extractor = await _build_extractor()

    async with get_session() as session:
        ctx = _IngestContext(
            session=session,
            extractor=extractor,
            quality_filter=YoutubeQualityFilter(await FilterSettingsRepository(session).get()),
            source_repo=SourceRepository(session),
            document_repo=DocumentRepository(session),
        )

        sources = await _save_sources(ctx, query, source_limit)
        await session.commit()

        saved_documents = await _save_documents(ctx, sources, document_limit)

    logger.info("Pipeline finished: %d source(s), %d new document(s)", len(sources), saved_documents)


async def reprocess_source(source_id: int, document_limit: int) -> None:
    """Re-run document extraction for a single, already-persisted source."""
    extractor = await _build_extractor()

    async with get_session() as session:
        ctx = _IngestContext(
            session=session,
            extractor=extractor,
            quality_filter=YoutubeQualityFilter(await FilterSettingsRepository(session).get()),
            source_repo=SourceRepository(session),
            document_repo=DocumentRepository(session),
        )

        item = await ctx.source_repo.get_with_document_count(source_id)
        if item is None:
            msg = f"Source {source_id} not found"
            raise ValueError(msg)

        saved_documents = await _save_documents(ctx, [item.source], document_limit)

    logger.info("Reprocessed source_id=%d: %d new document(s)", source_id, saved_documents)
