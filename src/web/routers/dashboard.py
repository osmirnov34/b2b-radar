from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from src.infrastructure.db.repositories import DocumentRepository, SourceRepository, YoutubeApiKeyRepository
from src.web.dependencies import SessionDep
from src.web.nav import nav_context
from src.web.templating import templates

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request, session: SessionDep) -> HTMLResponse:
    source_repo = SourceRepository(session)
    document_repo = DocumentRepository(session)
    api_key_repo = YoutubeApiKeyRepository(session)

    since = datetime.now(UTC) - timedelta(days=7)

    sources_total = await source_repo.count_all()
    sources_new = await source_repo.count_since(since)
    processed_sources = await document_repo.count_distinct_sources_with_documents()
    documents_total = await document_repo.count_all()
    documents_new = await document_repo.count_since(since)
    api_keys = await api_key_repo.list_all()
    active_keys = [key for key in api_keys if key.is_active]

    recent_sources, _ = await source_repo.list_paginated(search=None, status="all", page=1, page_size=5)

    context = {
        "request": request,
        "sources_total": sources_total,
        "sources_new": sources_new,
        "processed_sources": processed_sources,
        "documents_total": documents_total,
        "documents_new": documents_new,
        "active_keys_count": len(active_keys),
        "total_keys_count": len(api_keys),
        "recent_sources": recent_sources,
        **await nav_context(session),
    }
    return templates.TemplateResponse(request, "dashboard.html", context)
