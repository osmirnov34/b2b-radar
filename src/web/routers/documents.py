import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Literal

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from src.infrastructure.db.repositories import DocumentRepository
from src.infrastructure.db.session import get_session
from src.web.dependencies import SessionDep
from src.web.nav import nav_context
from src.web.templating import templates

router = APIRouter()

PAGE_SIZE = 20
_PERIOD_DAYS: dict[str, int | None] = {"all": None, "today": 1, "week": 7, "month": 30}


@router.get("/comments", response_class=HTMLResponse)
async def list_documents(
    request: Request,
    session: SessionDep,
    q: str | None = None,
    period: Literal["all", "today", "week", "month"] = "all",
    order: Literal["newest", "oldest"] = "newest",
    page: int = 1,
) -> HTMLResponse:
    document_repo = DocumentRepository(session)

    days = _PERIOD_DAYS[period]
    since = datetime.now(UTC) - timedelta(days=days) if days is not None else None

    items, total = await document_repo.list_paginated(
        search=q,
        source_id=None,
        since=since,
        order=order,
        page=page,
        page_size=PAGE_SIZE,
    )

    last_24h = await document_repo.count_since(datetime.now(UTC) - timedelta(days=1))
    sources_with_comments = await document_repo.count_distinct_sources_with_documents()

    context = {
        "request": request,
        "items": items,
        "total": total,
        "page": page,
        "page_size": PAGE_SIZE,
        "q": q or "",
        "period": period,
        "order": order,
        "total_all": await document_repo.count_all(),
        "last_24h": last_24h,
        "sources_with_comments": sources_with_comments,
        **await nav_context(session),
    }
    return templates.TemplateResponse(request, "documents.html", context)


@router.get("/comments/export.jsonl")
async def export_documents_jsonl(
    q: str | None = None,
    period: Literal["all", "today", "week", "month"] = "all",
) -> StreamingResponse:
    days = _PERIOD_DAYS[period]
    since = datetime.now(UTC) - timedelta(days=days) if days is not None else None

    async def generate() -> AsyncIterator[bytes]:
        async with get_session() as session:
            document_repo = DocumentRepository(session)
            async for document, source in document_repo.stream_for_export(search=q, since=since):
                row = {
                    "video_title": source.name,
                    "video_description": source.metadata_data.get("description"),
                    "video_channel": source.metadata_data.get("channel_title"),
                    "video_url": source.url,
                    "search_query": source.metadata_data.get("search_query"),
                    "comment_id": document.external_id,
                    "comment_text": document.text,
                    "comment_author": document.metadata_data.get("author_display_name"),
                    "comment_like_count": document.metadata_data.get("like_count"),
                    "comment_published_at": document.created_at.isoformat(),
                }
                yield (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")

    filename = f"b2b-radar-comments-{datetime.now(UTC):%Y%m%d-%H%M%S}.jsonl"
    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
