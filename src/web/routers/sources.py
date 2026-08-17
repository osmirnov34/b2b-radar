from typing import Annotated, Literal

from fastapi import APIRouter, BackgroundTasks, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from src.infrastructure.db.repositories import DocumentRepository, SourceRepository, YoutubeApiKeyRepository
from src.ingestion.pipeline import reprocess_failed_sources, reprocess_source, run
from src.web.dependencies import SessionDep
from src.web.nav import nav_context
from src.web.templating import templates

router = APIRouter()

PAGE_SIZE = 20

NO_ACTIVE_KEYS_ERROR = "no_active_keys"
EMPTY_QUERY_ERROR = "empty_query"


def _parse_queries(raw: str) -> list[str]:
    """Split a textarea value into distinct, trimmed search queries (one per line)."""
    seen: set[str] = set()
    queries: list[str] = []
    for line in raw.splitlines():
        query = line.strip()
        if query and query not in seen:
            seen.add(query)
            queries.append(query)
    return queries


@router.get("/videos", response_class=HTMLResponse)
async def list_sources(
    request: Request,
    session: SessionDep,
    q: str | None = None,
    status: Literal["all", "processed", "pending", "failed"] = "all",
    page: int = 1,
    error: str | None = None,
) -> HTMLResponse:
    source_repo = SourceRepository(session)
    items, total = await source_repo.list_paginated(search=q, status=status, page=page, page_size=PAGE_SIZE)

    context = {
        "request": request,
        "items": items,
        "total": total,
        "page": page,
        "page_size": PAGE_SIZE,
        "q": q or "",
        "status": status,
        "error": error,
        "failed_count": await source_repo.count_failed(),
        **await nav_context(session),
    }
    return templates.TemplateResponse(request, "sources.html", context)


@router.post("/videos")
async def add_source(
    background_tasks: BackgroundTasks,
    session: SessionDep,
    query: Annotated[str, Form()],
    source_limit: Annotated[int, Form()] = 50,
    document_limit: Annotated[int, Form()] = 100,
) -> RedirectResponse:
    queries = _parse_queries(query)
    if not queries:
        return RedirectResponse(url=f"/videos?error={EMPTY_QUERY_ERROR}", status_code=303)

    active_keys = await YoutubeApiKeyRepository(session).list_active_keys()
    if not active_keys:
        return RedirectResponse(url=f"/videos?error={NO_ACTIVE_KEYS_ERROR}", status_code=303)

    for single_query in queries:
        background_tasks.add_task(run, single_query, source_limit, document_limit)
    return RedirectResponse(url="/videos", status_code=303)


@router.get("/queries", response_class=HTMLResponse)
async def list_queries(request: Request, session: SessionDep) -> HTMLResponse:
    queries = await SourceRepository(session).list_search_queries()
    context = {
        "request": request,
        "queries": queries,
        **await nav_context(session),
    }
    return templates.TemplateResponse(request, "queries.html", context)


@router.get("/videos/{source_id}", response_class=HTMLResponse)
async def source_detail(
    request: Request,
    session: SessionDep,
    source_id: int,
    q: str | None = None,
    page: int = 1,
    error: str | None = None,
) -> HTMLResponse:
    source_repo = SourceRepository(session)
    item = await source_repo.get_with_document_count(source_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Source not found")

    document_repo = DocumentRepository(session)
    documents, total = await document_repo.list_for_source(source_id, search=q, page=page, page_size=PAGE_SIZE)

    context = {
        "request": request,
        "item": item,
        "documents": documents,
        "total": total,
        "page": page,
        "error": error,
        "page_size": PAGE_SIZE,
        "q": q or "",
        **await nav_context(session),
    }
    return templates.TemplateResponse(request, "source_detail.html", context)


@router.post("/videos/{source_id}/reprocess")
async def reprocess(
    background_tasks: BackgroundTasks,
    session: SessionDep,
    source_id: int,
    document_limit: Annotated[int, Form()] = 100,
) -> RedirectResponse:
    active_keys = await YoutubeApiKeyRepository(session).list_active_keys()
    if not active_keys:
        return RedirectResponse(url=f"/videos/{source_id}?error={NO_ACTIVE_KEYS_ERROR}", status_code=303)
    background_tasks.add_task(reprocess_source, source_id, document_limit)
    return RedirectResponse(url=f"/videos/{source_id}", status_code=303)


@router.post("/videos/retry-failed")
async def retry_failed(
    background_tasks: BackgroundTasks,
    session: SessionDep,
    document_limit: Annotated[int, Form()] = 100,
) -> RedirectResponse:
    """Queue a retry of every source stuck in FAILED (one background task loops them sequentially)."""
    active_keys = await YoutubeApiKeyRepository(session).list_active_keys()
    if not active_keys:
        return RedirectResponse(url=f"/videos?status=failed&error={NO_ACTIVE_KEYS_ERROR}", status_code=303)
    background_tasks.add_task(reprocess_failed_sources, document_limit)
    return RedirectResponse(url="/videos?status=failed", status_code=303)
