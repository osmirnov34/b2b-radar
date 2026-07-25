from typing import Annotated

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from src.domain.filter_settings import FilterSettings
from src.infrastructure.db.repositories import FilterSettingsRepository
from src.web.dependencies import SessionDep
from src.web.nav import nav_context
from src.web.templating import templates

router = APIRouter()

NonNegative = Annotated[int, Form(ge=0)]


@router.get("/settings", response_class=HTMLResponse)
async def show_settings(request: Request, session: SessionDep) -> HTMLResponse:
    settings = await FilterSettingsRepository(session).get()

    context = {
        "request": request,
        "settings": settings,
        **await nav_context(session),
    }
    return templates.TemplateResponse(request, "settings.html", context)


@router.post("/settings")
async def save_settings(
    session: SessionDep,
    *,
    document_min_likes: NonNegative = 0,
    document_min_length: NonNegative = 0,
    document_min_replies: NonNegative = 0,
) -> RedirectResponse:
    settings = FilterSettings(
        document_min_likes=document_min_likes,
        document_min_length=document_min_length,
        document_min_replies=document_min_replies,
    )
    await FilterSettingsRepository(session).update(settings)
    await session.commit()
    return RedirectResponse(url="/settings", status_code=303)
