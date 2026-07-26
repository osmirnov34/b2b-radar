import json
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse

from src.infrastructure.db.repositories import AnalysisRepository
from src.web.dependencies import SessionDep
from src.web.nav import nav_context
from src.web.templating import templates

router = APIRouter()

PAGE_SIZE = 50
EMPTY_FILE_ERROR = "empty_file"


def _parse_clusters(raw: bytes) -> list[dict[str, Any]]:
    """Parse a clusters.jsonl upload: one JSON object per line, keeping only cluster records."""
    clusters: list[dict[str, Any]] = []
    for line in raw.decode("utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("comments"), list):
            clusters.append(obj)
    return clusters


@router.get("/analysis", response_class=HTMLResponse)
async def list_runs(request: Request, session: SessionDep, error: str | None = None) -> HTMLResponse:
    runs = await AnalysisRepository(session).list_runs()
    context = {"request": request, "runs": runs, "error": error, **await nav_context(session)}
    return templates.TemplateResponse(request, "analysis_runs.html", context)


@router.post("/analysis")
async def upload_run(
    session: SessionDep,
    file: Annotated[UploadFile, File()],
    label: Annotated[str, Form()] = "",
) -> RedirectResponse:
    clusters = _parse_clusters(await file.read())
    if not clusters:
        return RedirectResponse(url=f"/analysis?error={EMPTY_FILE_ERROR}", status_code=303)

    run_label = label.strip() or (file.filename or "clusters.jsonl")
    run_id = await AnalysisRepository(session).create_run(run_label, clusters)
    await session.commit()
    return RedirectResponse(url=f"/analysis/{run_id}", status_code=303)


@router.post("/analysis/{run_id}/delete")
async def delete_run(session: SessionDep, run_id: int) -> RedirectResponse:
    await AnalysisRepository(session).delete_run(run_id)
    await session.commit()
    return RedirectResponse(url="/analysis", status_code=303)


@router.get("/analysis/{run_id}", response_class=HTMLResponse)
async def run_detail(request: Request, session: SessionDep, run_id: int) -> HTMLResponse:
    repo = AnalysisRepository(session)
    run = await repo.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    clusters = await repo.list_cluster_summaries(run_id)
    max_comments = max((c.n_comments for c in clusters), default=1)
    context = {
        "request": request,
        "run": run,
        "clusters": clusters,
        "max_comments": max_comments,
        **await nav_context(session),
    }
    return templates.TemplateResponse(request, "analysis_run.html", context)


@router.get("/analysis/{run_id}/clusters/{cluster_id}", response_class=HTMLResponse)
async def cluster_detail(
    request: Request,
    session: SessionDep,
    run_id: int,
    cluster_id: int,
    q: str | None = None,
    page: int = 1,
) -> HTMLResponse:
    repo = AnalysisRepository(session)
    run = await repo.get_run(run_id)
    cluster = await repo.get_cluster(cluster_id)
    if run is None or cluster is None or cluster.run_id != run_id:
        raise HTTPException(status_code=404, detail="Cluster not found")

    comments = cluster.comments or []
    if q:
        needle = q.lower()
        comments = [c for c in comments if needle in (c.get("text") or "").lower()]

    total = len(comments)
    page = max(1, page)
    start = (page - 1) * PAGE_SIZE
    page_comments = comments[start : start + PAGE_SIZE]

    context = {
        "request": request,
        "run": run,
        "cluster": cluster,
        "comments": page_comments,
        "total": total,
        "page": page,
        "page_size": PAGE_SIZE,
        "q": q or "",
        **await nav_context(session),
    }
    return templates.TemplateResponse(request, "analysis_cluster.html", context)
