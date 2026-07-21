import logging

from src.domain.source import Source
from src.infrastructure.api.youtube import YoutubeClient
from src.infrastructure.db.repositories import DocumentRepository, SourceRepository, YoutubeApiKeyRepository
from src.infrastructure.db.session import get_session
from src.infrastructure.extractor.youtube import YoutubeExtractor

logger = logging.getLogger(__name__)


async def _save_sources(
    extractor: YoutubeExtractor,
    source_repo: SourceRepository,
    query: str,
    limit: int,
) -> list[Source]:
    sources = extractor.extract_sources(query, limit)
    saved_sources = []

    for source in sources:
        existing = await source_repo.get_by_url(source.url)
        if existing is not None:
            saved_sources.append(existing)
            continue
        source.metadata["search_query"] = query
        saved_sources.append(await source_repo.add(source))

    return saved_sources


async def _save_documents(
    extractor: YoutubeExtractor,
    document_repo: DocumentRepository,
    sources: list[Source],
    limit: int,
) -> int:
    saved_count = 0

    for source in sources:
        documents = extractor.extract_documents(source, limit)
        if not documents:
            continue

        existing_ids = await document_repo.list_existing_external_ids([document.external_id for document in documents])
        new_documents = [document for document in documents if document.external_id not in existing_ids]

        if new_documents:
            await document_repo.add_many(new_documents)
            saved_count += len(new_documents)

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
        source_repo = SourceRepository(session)
        document_repo = DocumentRepository(session)

        sources = await _save_sources(extractor, source_repo, query, source_limit)
        await session.commit()

        saved_documents = await _save_documents(extractor, document_repo, sources, document_limit)
        await session.commit()

    logger.info("Pipeline finished: %d source(s), %d new document(s)", len(sources), saved_documents)


async def reprocess_source(source_id: int, document_limit: int) -> None:
    """Re-run document extraction for a single, already-persisted source."""
    extractor = await _build_extractor()

    async with get_session() as session:
        source_repo = SourceRepository(session)
        document_repo = DocumentRepository(session)

        item = await source_repo.get_with_document_count(source_id)
        if item is None:
            msg = f"Source {source_id} not found"
            raise ValueError(msg)

        saved_documents = await _save_documents(extractor, document_repo, [item.source], document_limit)
        await session.commit()

    logger.info("Reprocessed source_id=%d: %d new document(s)", source_id, saved_documents)
