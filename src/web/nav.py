from sqlalchemy.ext.asyncio import AsyncSession

from src.infrastructure.db.repositories import DocumentRepository, SourceRepository


async def nav_context(session: AsyncSession) -> dict[str, int]:
    """Sidebar counters shown on every page (sources and documents totals)."""
    return {
        "nav_sources_count": await SourceRepository(session).count_all(),
        "nav_documents_count": await DocumentRepository(session).count_all(),
    }
