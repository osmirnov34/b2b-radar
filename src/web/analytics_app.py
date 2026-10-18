"""Independent read-only analytics API with separate private-example access."""

from __future__ import annotations

import secrets
from datetime import date  # noqa: TC003 -- Pydantic resolves response field types at runtime.
from typing import TYPE_CHECKING, Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import get_analytics_api_settings
from src.infrastructure.db.session import get_session
from src.operations.problem_priorities import load_problem_priorities

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable
    from typing import Any

_bearer = HTTPBearer(auto_error=False)
Credentials = Annotated[HTTPAuthorizationCredentials | None, Security(_bearer)]
PageLimit = Annotated[int, Query(ge=1, le=200)]
PageOffset = Annotated[int, Query(ge=0)]
PositiveID = Annotated[int, Query(ge=1)]


async def require_analytics_access(credentials: Credentials) -> None:
    """Require an aggregate credential before opening a database session."""
    settings = get_analytics_api_settings()
    configured = settings.analytics_api_token
    if configured is None or not configured.get_secret_value().strip():
        raise HTTPException(status_code=503, detail="Analytics API access is not configured")
    if credentials is None or not secrets.compare_digest(
        credentials.credentials.encode(),
        configured.get_secret_value().encode(),
    ):
        raise HTTPException(
            status_code=401, detail="Invalid analytics credential", headers={"WWW-Authenticate": "Bearer"}
        )


async def require_private_access(credentials: Credentials) -> None:
    """Require a separate secret for restricted source text."""
    settings = get_analytics_api_settings()
    configured = settings.analytics_private_token
    aggregate = settings.analytics_api_token
    if (
        configured is None
        or not configured.get_secret_value().strip()
        or (aggregate is not None and configured.get_secret_value() == aggregate.get_secret_value())
    ):
        raise HTTPException(status_code=503, detail="Private example access is not configured")
    if credentials is None or not secrets.compare_digest(
        credentials.credentials.encode(),
        configured.get_secret_value().encode(),
    ):
        raise HTTPException(
            status_code=401, detail="Invalid private credential", headers={"WWW-Authenticate": "Bearer"}
        )


async def get_analytics_session() -> AsyncIterator[AsyncSession]:
    """Enforce PostgreSQL read-only transactions and sanitize database failures."""
    try:
        async with get_session() as session, session.begin():
            await session.execute(text("SET TRANSACTION READ ONLY"))
            yield session
    except SQLAlchemyError:
        raise HTTPException(status_code=503, detail="Analytical storage is unavailable") from None


Session = Annotated[AsyncSession, Depends(get_analytics_session)]


class _Response(BaseModel):
    model_config = ConfigDict(extra="ignore")


class DatasetSummary(_Response):
    dataset_id: int
    source: str
    feedback_count: int
    organization_count: int
    published_on_min: date | None
    published_on_max: date | None


class OrganizationKPI(_Response):
    dataset_id: int
    organization_id: str
    organization_name: str
    organization_url: str
    feedback_count: int
    average_rating: float
    low_rating_share: float
    high_rating_share: float
    rating_1_count: int
    rating_2_count: int
    rating_3_count: int
    rating_4_count: int
    rating_5_count: int
    sample_sufficient: bool


class ProblemKPI(_Response):
    run_id: int
    dataset_id: int
    organization_id: str
    category_id: str
    category_name_ru: str
    feedback_count: int
    complaint_count: int
    mixed_count: int
    praise_count: int
    complaint_share: float
    mixed_share: float
    praise_share: float
    accepted_complaint_count: int
    unreviewed_complaint_count: int
    mean_category_probability: float | None
    sample_sufficient: bool


class MonthlyKPI(_Response):
    dataset_id: int
    organization_id: str
    month: date
    feedback_count: int
    average_rating: float
    low_rating_share: float
    sample_sufficient: bool


class ProblemMonthlyKPI(_Response):
    run_id: int
    dataset_id: int
    organization_id: str
    category_id: str
    month: date
    feedback_count: int
    complaint_count: int
    mixed_count: int
    praise_count: int
    complaint_share: float
    accepted_complaint_count: int
    unreviewed_complaint_count: int
    sample_sufficient: bool


class PrivateExample(_Response):
    feedback_id: str
    organization_id: str
    category_id: str
    text: str
    organization_url: str
    stance: str
    review_status: str


async def _require_dataset(session: AsyncSession, dataset_id: int) -> None:
    exists = await session.scalar(text("SELECT id FROM analytics.datasets WHERE id = :id"), {"id": dataset_id})
    if exists is None:
        raise HTTPException(status_code=404, detail="Dataset not found")


async def _require_run(session: AsyncSession, run_id: int) -> None:
    exists = await session.scalar(text("SELECT id FROM analytics.problem_runs WHERE id = :id"), {"id": run_id})
    if exists is None:
        raise HTTPException(status_code=404, detail="Problem run not found")


async def dataset_summary(session: Session, dataset_id: PositiveID) -> DatasetSummary:
    """Describe one exact immutable feedback snapshot."""
    result = await session.execute(
        text("""
        SELECT id AS dataset_id, source, record_count AS feedback_count, organization_count,
               published_on_min, published_on_max FROM analytics.datasets WHERE id = :id
    """),
        {"id": dataset_id},
    )
    row = result.mappings().one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return DatasetSummary.model_validate(dict(row))


async def organizations(
    session: Session,
    dataset_id: PositiveID,
    limit: PageLimit = 50,
    offset: PageOffset = 0,
) -> list[OrganizationKPI]:
    """List organization KPIs in deterministic pages."""
    await _require_dataset(session, dataset_id)
    rows = await session.execute(
        text("""
        SELECT * FROM analytics.organization_kpis WHERE dataset_id = :id
        ORDER BY organization_id LIMIT :limit OFFSET :offset
    """),
        {"id": dataset_id, "limit": limit, "offset": offset},
    )
    return [OrganizationKPI.model_validate(dict(row)) for row in rows.mappings()]


async def problems(
    session: Session,
    run_id: PositiveID,
    organization_id: str | None = None,
    limit: PageLimit = 50,
    offset: PageOffset = 0,
) -> list[ProblemKPI]:
    """Read corrected category metrics for one selected assessment run."""
    await _require_run(session, run_id)
    rows = await session.execute(
        text("""
        SELECT * FROM analytics.problem_kpis WHERE run_id = :id
          AND (CAST(:org AS text) IS NULL OR organization_id = :org)
        ORDER BY organization_id, category_id LIMIT :limit OFFSET :offset
    """),
        {"id": run_id, "org": organization_id, "limit": limit, "offset": offset},
    )
    return [ProblemKPI.model_validate(dict(row)) for row in rows.mappings()]


async def monthly(
    session: Session,
    dataset_id: PositiveID,
    organization_id: str | None = None,
    limit: PageLimit = 50,
    offset: PageOffset = 0,
) -> list[MonthlyKPI]:
    """Read monthly organization metrics; missing months remain absent."""
    await _require_dataset(session, dataset_id)
    rows = await session.execute(
        text("""
        SELECT * FROM analytics.organization_monthly_kpis WHERE dataset_id = :id
          AND (CAST(:org AS text) IS NULL OR organization_id = :org)
        ORDER BY month, organization_id LIMIT :limit OFFSET :offset
    """),
        {"id": dataset_id, "org": organization_id, "limit": limit, "offset": offset},
    )
    return [MonthlyKPI.model_validate(dict(row)) for row in rows.mappings()]


async def problem_monthly(
    session: Session,
    run_id: PositiveID,
    organization_id: str | None = None,
    limit: PageLimit = 50,
    offset: PageOffset = 0,
) -> list[ProblemMonthlyKPI]:
    """Read monthly category signals from the corrected view."""
    await _require_run(session, run_id)
    rows = await session.execute(
        text("""
        SELECT * FROM analytics.problem_monthly_kpis WHERE run_id = :id
          AND (CAST(:org AS text) IS NULL OR organization_id = :org)
        ORDER BY month, organization_id, category_id LIMIT :limit OFFSET :offset
    """),
        {"id": run_id, "org": organization_id, "limit": limit, "offset": offset},
    )
    return [ProblemMonthlyKPI.model_validate(dict(row)) for row in rows.mappings()]


async def priorities(
    session: Session,
    run_id: PositiveID,
    limit: PageLimit = 50,
    offset: PageOffset = 0,
) -> list[dict[str, object]]:
    """Return ranked recommendations while keeping evidence identifiers private."""
    await _require_run(session, run_id)
    ranked = await load_problem_priorities(session, run_id)
    return [row.model_dump(exclude={"signal": {"evidence_feedback_ids"}}) for row in ranked[offset : offset + limit]]


async def private_examples(  # noqa: PLR0913 -- explicit API filters and pagination.
    session: Session,
    run_id: PositiveID,
    category_id: str,
    *,
    organization_id: str | None = None,
    limit: PageLimit = 5,
    offset: PageOffset = 0,
) -> list[PrivateExample]:
    """Read accepted examples only after separate private authorization."""
    await _require_run(session, run_id)
    rows = await session.execute(
        text("""
        SELECT pa.feedback_id, f.organization_id, pa.category_id, f.text, o.organization_url,
               pa.stance, pa.review_status
        FROM analytics.problem_assessments pa
        JOIN analytics.feedback f ON f.dataset_id = pa.dataset_id AND f.feedback_id = pa.feedback_id
        JOIN analytics.organizations o ON o.dataset_id = f.dataset_id AND o.organization_id = f.organization_id
        WHERE pa.run_id = :id AND pa.category_id = :category AND pa.review_status = 'accepted'
          AND (CAST(:org AS text) IS NULL OR f.organization_id = :org)
        ORDER BY pa.category_probability DESC, pa.feedback_id LIMIT :limit OFFSET :offset
    """),
        {"id": run_id, "category": category_id, "org": organization_id, "limit": limit, "offset": offset},
    )
    return [PrivateExample.model_validate(dict(row)) for row in rows.mappings()]


def create_analytics_app() -> FastAPI:
    """Build an independent API without ingestion startup hooks or write routes."""
    application = FastAPI(title="B2B Radar Analytics", version="1.0.0")
    aggregate_routes: dict[str, Callable[..., Any]] = {
        "/api/analytics/summary": dataset_summary,
        "/api/analytics/organizations": organizations,
        "/api/analytics/problems": problems,
        "/api/analytics/monthly": monthly,
        "/api/analytics/problem-monthly": problem_monthly,
        "/api/analytics/priorities": priorities,
    }
    for path, endpoint in aggregate_routes.items():
        application.add_api_route(path, endpoint, methods=["GET"], dependencies=[Depends(require_analytics_access)])
    application.add_api_route(
        "/api/analytics/private/examples",
        private_examples,
        methods=["GET"],
        dependencies=[Depends(require_private_access)],
    )
    return application


app = create_analytics_app()
