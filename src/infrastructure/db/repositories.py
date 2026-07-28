from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from sqlalchemy import case, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.domain.api_key import ApiKey
from src.domain.document import Document
from src.domain.source import IngestStatus, Source, SourceType
from src.infrastructure.db.models import (
    AnalysisRunModel,
    ClusterModel,
    DocumentModel,
    SourceModel,
    YoutubeApiKeyModel,
)

SourceStatus = Literal["all", "processed", "pending", "failed"]


@dataclass
class SourceListItem:
    """A source enriched with its extracted document count, for list/detail views."""

    source: Source
    document_count: int

    @property
    def status(self) -> Literal["processed", "pending"]:
        return "processed" if self.document_count > 0 else "pending"


@dataclass
class SearchQueryItem:
    """A distinct search query with how many sources it found and when it was last used."""

    query: str
    source_count: int
    last_used: datetime


@dataclass
class DocumentListItem:
    """A document enriched with its parent source name, for cross-source list views."""

    document: Document
    source_name: str


class SourceRepository:
    """Persists and retrieves Source domain objects."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_by_url(self, url: str) -> Source | None:
        model = await self.session.scalar(select(SourceModel).where(SourceModel.url == url))
        if model is None:
            return None
        return self._to_domain(model)

    async def get_with_document_count(self, source_id: int) -> SourceListItem | None:
        row = (
            await self.session.execute(
                select(SourceModel, func.count(DocumentModel.id))
                .outerjoin(DocumentModel, DocumentModel.source_id == SourceModel.id)
                .where(SourceModel.id == source_id)
                .group_by(SourceModel.id),
            )
        ).first()
        if row is None:
            return None
        model, document_count = row
        return SourceListItem(source=self._to_domain(model), document_count=document_count)

    async def list_paginated(
        self,
        *,
        search: str | None,
        status: SourceStatus,
        page: int,
        page_size: int,
    ) -> tuple[list[SourceListItem], int]:
        document_count = func.count(DocumentModel.id)
        base_query = select(SourceModel, document_count).outerjoin(
            DocumentModel,
            DocumentModel.source_id == SourceModel.id,
        )

        if search:
            pattern = f"%{search}%"
            base_query = base_query.where(
                or_(SourceModel.name.ilike(pattern), SourceModel.metadata_data["channel_title"].astext.ilike(pattern)),
            )

        if status == "failed":
            base_query = base_query.where(SourceModel.ingest_status == IngestStatus.FAILED)

        base_query = base_query.group_by(SourceModel.id)
        if status == "processed":
            base_query = base_query.having(document_count > 0)
        elif status == "pending":
            base_query = base_query.having(document_count == 0)

        total = await self.session.scalar(select(func.count()).select_from(base_query.subquery())) or 0

        rows = (
            await self.session.execute(
                base_query.order_by(SourceModel.extracted_at.desc()).offset((page - 1) * page_size).limit(page_size),
            )
        ).all()

        items = [SourceListItem(source=self._to_domain(model), document_count=count) for model, count in rows]
        return items, total

    async def count_all(self) -> int:
        return await self.session.scalar(select(func.count()).select_from(SourceModel)) or 0

    async def list_search_queries(self) -> list["SearchQueryItem"]:
        """Distinct search queries used to discover sources, newest first.

        Reads ``metadata_data->>'search_query'`` (stamped in the pipeline). A source's query is the
        first one that found it (dedup by URL), so per-query counts can undercount overlapping
        queries, and queries that found nothing are absent — this is a history of what was searched.
        """
        query_text = SourceModel.metadata_data["search_query"].astext
        rows = (
            await self.session.execute(
                select(query_text, func.count(SourceModel.id), func.max(SourceModel.extracted_at))
                .where(query_text.isnot(None))
                .group_by(query_text)
                .order_by(func.max(SourceModel.extracted_at).desc()),
            )
        ).all()
        return [SearchQueryItem(query=q, source_count=count, last_used=last_used) for q, count, last_used in rows]

    async def count_failed(self) -> int:
        """How many sources are stuck in FAILED — drives the retry affordance in the UI."""
        return (
            await self.session.scalar(
                select(func.count()).select_from(SourceModel).where(SourceModel.ingest_status == IngestStatus.FAILED),
            )
            or 0
        )

    async def list_failed_ids(self) -> list[int]:
        """IDs of every source stuck in FAILED, for a bulk retry (CLI or UI)."""
        result = await self.session.scalars(
            select(SourceModel.id).where(SourceModel.ingest_status == IngestStatus.FAILED),
        )
        return list(result.all())

    async def reset_stale_running(self, older_than: timedelta) -> int:
        """Flip RUNNING sources stalled longer than `older_than` to FAILED. Call at process startup.

        A source is only RUNNING for the few minutes it takes to extract one video's comments
        (ingestion is sequential, so at most one is RUNNING per process at a time). Anything RUNNING
        far longer than that is orphaned — a crash/restart killed the coroutine before it could mark
        the source FAILED. We key off ingest_started_at instead of blindly resetting *all* RUNNING so
        a concurrent, still-alive ingest (e.g. a long CLI backfill running while the web app restarts)
        is never falsely failed: its in-flight source is recent, so it's skipped. NULL covers legacy
        rows written before ingest_started_at existed. Reset sources become visible + retryable via
        the same failed-sources UI/CLI.
        """
        cutoff = datetime.now(UTC) - older_than
        result = await self.session.execute(
            update(SourceModel)
            .where(
                SourceModel.ingest_status == IngestStatus.RUNNING,
                or_(SourceModel.ingest_started_at.is_(None), SourceModel.ingest_started_at < cutoff),
            )
            .values(ingest_status=IngestStatus.FAILED, ingest_error="Orphaned: process restarted mid-extraction")
            .returning(SourceModel.id),
        )
        return len(result.all())

    async def count_since(self, since: datetime) -> int:
        return (
            await self.session.scalar(
                select(func.count()).select_from(SourceModel).where(SourceModel.extracted_at >= since),
            )
            or 0
        )

    async def add(self, source: Source) -> Source:
        model = SourceModel(
            type=source.type,
            url=source.url,
            name=source.name,
            metadata_data=source.metadata,
        )
        self.session.add(model)
        await self.session.flush()
        return self._to_domain(model)

    async def set_ingest_status(self, source_id: int, status: IngestStatus, error: str | None = None) -> None:
        """Record the outcome of a source's document extraction so the UI can surface progress/failures."""
        model = await self.session.get(SourceModel, source_id)
        if model is None:
            msg = f"Source {source_id} not found"
            raise ValueError(msg)
        model.ingest_status = status
        model.ingest_error = error
        # Stamp the start of a RUNNING window so the startup reaper can age out stalled ingests.
        if status == IngestStatus.RUNNING:
            model.ingest_started_at = datetime.now(UTC)
        await self.session.flush()

    @staticmethod
    def _to_domain(model: SourceModel) -> Source:
        return Source(
            id=model.id,
            type=SourceType(model.type),
            url=model.url,
            name=model.name,
            metadata=model.metadata_data,
            ingest_status=IngestStatus(model.ingest_status),
            ingest_error=model.ingest_error,
            ingest_started_at=model.ingest_started_at,
            extracted_at=model.extracted_at,
        )


class DocumentRepository:
    """Persists and retrieves Document domain objects."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def list_existing_external_ids(self, external_ids: Sequence[str]) -> set[str]:
        if not external_ids:
            return set()
        rows = await self.session.scalars(
            select(DocumentModel.external_id).where(DocumentModel.external_id.in_(external_ids)),
        )
        return set(rows)

    async def add_many(self, documents: Sequence[Document]) -> list[Document]:
        models = [
            DocumentModel(
                source_id=document.source_id,
                external_id=document.external_id,
                text=document.text,
                created_at=document.created_at,
                metadata_data=document.metadata,
            )
            for document in documents
        ]
        self.session.add_all(models)
        await self.session.flush()
        return [self._to_domain(model) for model in models]

    async def list_paginated(
        self,
        *,
        search: str | None,
        source_id: int | None,
        since: datetime | None,
        order: Literal["newest", "oldest"],
        page: int,
        page_size: int,
    ) -> tuple[list[DocumentListItem], int]:
        base_query = select(DocumentModel, SourceModel.name).join(
            SourceModel,
            SourceModel.id == DocumentModel.source_id,
        )

        if search:
            base_query = base_query.where(DocumentModel.text.ilike(f"%{search}%"))
        if source_id is not None:
            base_query = base_query.where(DocumentModel.source_id == source_id)
        if since is not None:
            base_query = base_query.where(DocumentModel.created_at >= since)

        total = await self.session.scalar(select(func.count()).select_from(base_query.subquery())) or 0

        order_column = DocumentModel.created_at.desc() if order == "newest" else DocumentModel.created_at.asc()
        rows = (
            await self.session.execute(
                base_query.order_by(order_column).offset((page - 1) * page_size).limit(page_size),
            )
        ).all()

        items = [
            DocumentListItem(document=self._to_domain(model), source_name=source_name) for model, source_name in rows
        ]
        return items, total

    async def list_for_source(
        self,
        source_id: int,
        *,
        search: str | None,
        page: int,
        page_size: int,
    ) -> tuple[list[Document], int]:
        base_query = select(DocumentModel).where(DocumentModel.source_id == source_id)
        if search:
            base_query = base_query.where(DocumentModel.text.ilike(f"%{search}%"))

        total = await self.session.scalar(select(func.count()).select_from(base_query.subquery())) or 0
        rows = await self.session.scalars(
            base_query.order_by(DocumentModel.created_at.desc()).offset((page - 1) * page_size).limit(page_size),
        )
        return [self._to_domain(model) for model in rows], total

    async def count_all(self) -> int:
        return await self.session.scalar(select(func.count()).select_from(DocumentModel)) or 0

    async def count_since(self, since: datetime) -> int:
        return (
            await self.session.scalar(
                select(func.count()).select_from(DocumentModel).where(DocumentModel.extracted_at >= since),
            )
            or 0
        )

    async def count_distinct_sources_with_documents(self) -> int:
        return await self.session.scalar(select(func.count(func.distinct(DocumentModel.source_id)))) or 0

    async def stream_for_export(
        self,
        *,
        search: str | None,
        since: datetime | None,
    ) -> AsyncIterator[tuple[DocumentModel, SourceModel]]:
        """Yield (document, source) pairs matching the filters, without loading the whole result set at once."""
        query = select(DocumentModel, SourceModel).join(SourceModel, SourceModel.id == DocumentModel.source_id)

        if search:
            query = query.where(DocumentModel.text.ilike(f"%{search}%"))
        if since is not None:
            query = query.where(DocumentModel.created_at >= since)

        query = query.order_by(DocumentModel.created_at.desc())

        result = await self.session.stream(query)
        async for document, source in result:
            yield document, source

    @staticmethod
    def _to_domain(model: DocumentModel) -> Document:
        return Document(
            id=model.id,
            source_id=model.source_id,
            external_id=model.external_id,
            text=model.text,
            created_at=model.created_at,
            metadata=model.metadata_data,
            extracted_at=model.extracted_at,
        )


@dataclass
class ClusterSummary:
    """A cluster's card-level fields (no comments payload), for the run overview grid."""

    id: int
    topic_id: int
    n_comments: int
    n_authors: int
    n_channels: int
    keywords: list[str]


class AnalysisRepository:
    """Persists and retrieves uploaded clustering results (runs -> clusters -> comments)."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_run(self, label: str, clusters: list[dict[str, Any]]) -> int:
        """Persist a parsed clusters.jsonl as one run + its clusters. Returns the new run id.

        Run-level distinct authors/channels are counted across all comments (overlap-free),
        not summed from per-cluster counts (an author can recur across clusters).
        """
        authors: set[str] = set()
        channels: set[str] = set()
        total_comments = 0
        for c in clusters:
            for comment in c.get("comments", []):
                authors.add(comment.get("author", ""))
                channels.add(comment.get("channel", ""))
            total_comments += int(c.get("n_comments", len(c.get("comments", []))))

        run = AnalysisRunModel(
            label=label,
            n_clusters=len(clusters),
            n_comments=total_comments,
            n_authors=len(authors),
            n_channels=len(channels),
        )
        self.session.add(run)
        await self.session.flush()

        self.session.add_all(
            [
                ClusterModel(
                    run_id=run.id,
                    topic_id=int(c.get("topic_id", -1)),
                    n_comments=int(c.get("n_comments", len(c.get("comments", [])))),
                    n_authors=int(c.get("n_authors", 0)),
                    n_channels=int(c.get("n_channels", 0)),
                    keywords=c.get("keywords", []),
                    comments=c.get("comments", []),
                )
                for c in clusters
            ],
        )
        await self.session.flush()
        return run.id

    async def list_runs(self) -> list[AnalysisRunModel]:
        rows = await self.session.scalars(select(AnalysisRunModel).order_by(AnalysisRunModel.created_at.desc()))
        return list(rows)

    async def get_run(self, run_id: int) -> AnalysisRunModel | None:
        return await self.session.get(AnalysisRunModel, run_id)

    async def list_cluster_summaries(self, run_id: int) -> list[ClusterSummary]:
        """Cluster cards for a run, largest first — selects everything except the heavy comments blob."""
        rows = await self.session.execute(
            select(
                ClusterModel.id,
                ClusterModel.topic_id,
                ClusterModel.n_comments,
                ClusterModel.n_authors,
                ClusterModel.n_channels,
                ClusterModel.keywords,
            )
            .where(ClusterModel.run_id == run_id)
            .order_by(ClusterModel.n_comments.desc()),
        )
        return [
            ClusterSummary(
                id=r.id,
                topic_id=r.topic_id,
                n_comments=r.n_comments,
                n_authors=r.n_authors,
                n_channels=r.n_channels,
                keywords=r.keywords or [],
            )
            for r in rows
        ]

    async def get_cluster(self, cluster_id: int) -> ClusterModel | None:
        return await self.session.get(ClusterModel, cluster_id)

    async def delete_run(self, run_id: int) -> None:
        run = await self.session.get(AnalysisRunModel, run_id)
        if run is not None:
            await self.session.delete(run)  # clusters cascade via FK ON DELETE CASCADE
            await self.session.flush()

    async def count_all(self) -> int:
        return await self.session.scalar(select(func.count()).select_from(AnalysisRunModel)) or 0


class YoutubeApiKeyRepository:
    """Persists and retrieves managed YouTube API keys."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def list_active_keys(self) -> list[str]:
        rows = await self.session.scalars(
            select(YoutubeApiKeyModel.key).where(YoutubeApiKeyModel.is_active.is_(True)),
        )
        return list(rows)

    async def list_all(self) -> list[ApiKey]:
        rows = await self.session.scalars(
            select(YoutubeApiKeyModel).order_by(
                case((YoutubeApiKeyModel.is_active, 0), else_=1),
                YoutubeApiKeyModel.id,
            ),
        )
        return [self._to_domain(model) for model in rows]

    async def add(self, name: str | None, key: str) -> ApiKey:
        model = YoutubeApiKeyModel(name=name, key=key)
        self.session.add(model)
        await self.session.flush()
        return self._to_domain(model)

    async def set_active(self, key_id: int, *, is_active: bool) -> None:
        model = await self.session.get(YoutubeApiKeyModel, key_id)
        if model is None:
            msg = f"API key {key_id} not found"
            raise ValueError(msg)
        model.is_active = is_active
        await self.session.flush()

    @staticmethod
    def _to_domain(model: YoutubeApiKeyModel) -> ApiKey:
        return ApiKey(
            id=model.id,
            name=model.name,
            key=model.key,
            is_active=model.is_active,
            created_at=model.created_at,
        )
