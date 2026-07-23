from typing import Annotated, Literal

from fastapi import APIRouter, BackgroundTasks, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from src.infrastructure.db.repositories import DocumentRepository, SourceRepository
from src.pipeline import reprocess_source, run
from src.web.dependencies import SessionDep
from src.web.nav import nav_context
from src.web.templating import templates

router = APIRouter()

PAGE_SIZE = 20


@router.get("/videos", response_class=HTMLResponse)
async def list_sources(
    request: Request,
    session: SessionDep,
    q: str | None = None,
    status: Literal["all", "processed", "pending"] = "all",
    page: int = 1,
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
        **await nav_context(session),
    }
    return templates.TemplateResponse(request, "sources.html", context)


@router.post("/videos")
async def add_source(
    background_tasks: BackgroundTasks,
    query: Annotated[str, Form()],
    source_limit: Annotated[int, Form()] = 50,
    document_limit: Annotated[int, Form()] = 100,
) -> RedirectResponse:
    background_tasks.add_task(run, query, source_limit, document_limit)
    return RedirectResponse(url="/videos", status_code=303)


@router.get("/videos/{source_id}", response_class=HTMLResponse)
async def source_detail(
    request: Request,
    session: SessionDep,
    source_id: int,
    q: str | None = None,
    page: int = 1,
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
        "page_size": PAGE_SIZE,
        "q": q or "",
        **await nav_context(session),
    }
    return templates.TemplateResponse(request, "source_detail.html", context)


@router.post("/videos/{source_id}/reprocess")
async def reprocess(
    background_tasks: BackgroundTasks,
    source_id: int,
    document_limit: Annotated[int, Form()] = 100,
) -> RedirectResponse:
    background_tasks.add_task(reprocess_source, source_id, document_limit)
    return RedirectResponse(url=f"/videos/{source_id}", status_code=303)
