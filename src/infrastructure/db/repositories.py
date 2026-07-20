from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.domain.document import Document
from src.domain.source import Source, SourceType
from src.infrastructure.db.models import DocumentModel, SourceModel, YoutubeApiKeyModel


class SourceRepository:
    """Persists and retrieves Source domain objects."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_by_url(self, url: str) -> Source | None:
        model = await self.session.scalar(select(SourceModel).where(SourceModel.url == url))
        if model is None:
            return None
        return self._to_domain(model)

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

    @staticmethod
    def _to_domain(model: SourceModel) -> Source:
        return Source(
            id=model.id,
            type=SourceType(model.type),
            url=model.url,
            name=model.name,
            metadata=model.metadata_data,
        )


class DocumentRepository:
    """Persists and retrieves Document domain objects."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_existing_external_ids(self, external_ids: Sequence[str]) -> set[str]:
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

    @staticmethod
    def _to_domain(model: DocumentModel) -> Document:
        return Document(
            id=model.id,
            source_id=model.source_id,
            external_id=model.external_id,
            text=model.text,
            created_at=model.created_at,
            metadata=model.metadata_data,
        )


class YoutubeApiKeyRepository:
    """Reads YouTube API keys managed by an admin."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_active_keys(self) -> list[str]:
        rows = await self.session.scalars(
            select(YoutubeApiKeyModel.key).where(YoutubeApiKeyModel.is_active.is_(True)),
        )
        return list(rows)
