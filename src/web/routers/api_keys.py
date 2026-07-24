from typing import Annotated

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.exc import IntegrityError

from src.infrastructure.db.repositories import YoutubeApiKeyRepository
from src.web.dependencies import SessionDep
from src.web.nav import nav_context
from src.web.templating import templates

router = APIRouter()

DUPLICATE_KEY_ERROR = "duplicate_key"


@router.get("/api-keys", response_class=HTMLResponse)
async def list_api_keys(request: Request, session: SessionDep, error: str | None = None) -> HTMLResponse:
    repo = YoutubeApiKeyRepository(session)
    keys = await repo.list_all()

    context = {
        "request": request,
        "keys": keys,
        "active_count": sum(1 for key in keys if key.is_active),
        "error": error,
        **await nav_context(session),
    }
    return templates.TemplateResponse(request, "api_keys.html", context)


@router.post("/api-keys")
async def add_api_key(
    session: SessionDep,
    name: Annotated[str, Form()],
    key: Annotated[str, Form()],
) -> RedirectResponse:
    try:
        await YoutubeApiKeyRepository(session).add(name or None, key)
        await session.commit()
    except IntegrityError:
        await session.rollback()
        return RedirectResponse(url=f"/api-keys?error={DUPLICATE_KEY_ERROR}", status_code=303)
    return RedirectResponse(url="/api-keys", status_code=303)


@router.post("/api-keys/{key_id}/toggle")
async def toggle_api_key(
    session: SessionDep,
    key_id: int,
    is_active: Annotated[bool, Form()],
) -> RedirectResponse:
    await YoutubeApiKeyRepository(session).set_active(key_id, is_active=is_active)
    await session.commit()
    return RedirectResponse(url="/api-keys", status_code=303)
