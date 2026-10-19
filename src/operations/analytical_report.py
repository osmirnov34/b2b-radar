"""Read one consistent aggregate snapshot for a management PDF report."""

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import text

from src.analytics.report import (
    AnalyticalReport,
    ReportCategory,
    ReportMonth,
    ReportOrganization,
    ReportRecommendation,
)
from src.operations.problem_priorities import load_problem_priorities

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


async def load_analytical_report(session: "AsyncSession", run_id: int) -> AnalyticalReport:
    """Load validated aggregates in one repeatable-read, read-only transaction."""
    if run_id < 1 or session.in_transaction():
        msg = "report loader requires a positive run_id and a fresh session"
        raise ValueError(msg)
    async with session.begin():
        await session.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
        result = await session.execute(
            text("""
            SELECT r.id AS run_id, r.dataset_id, d.artifact_sha256 AS dataset_sha256,
                   r.artifact_sha256 AS assessments_sha256, r.taxonomy_id, r.taxonomy_version,
                   r.ruleset_version, d.source, r.generated_at AS run_generated_at,
                   d.record_count AS feedback_count, d.published_on_min, d.published_on_max
            FROM analytics.problem_runs r JOIN analytics.datasets d ON d.id = r.dataset_id WHERE r.id = :run
        """),
            {"run": run_id},
        )
        metadata = result.mappings().one_or_none()
        if metadata is None:
            msg = "selected analytical run does not exist"
            raise ValueError(msg)
        organizations = await session.execute(
            text("""
            SELECT organization_id, organization_name, feedback_count, rating_sum, low_rating_count
            FROM analytics_bi.organization_overview WHERE run_id = :run ORDER BY organization_id
        """),
            {"run": run_id},
        )
        categories = await session.execute(
            text("""
            SELECT category_id, category_name_ru, SUM(feedback_count) AS feedback_count,
                   SUM(complaint_count) AS complaint_count, SUM(mixed_count) AS mixed_count,
                   SUM(praise_count) AS praise_count, SUM(accepted_complaint_count) AS accepted_complaint_count,
                   SUM(unreviewed_complaint_count) AS unreviewed_complaint_count, SUM(rejected_count) AS rejected_count
            FROM analytics_bi.category_overview WHERE run_id = :run
            GROUP BY category_id, category_name_ru ORDER BY category_id
        """),
            {"run": run_id},
        )
        months = await session.execute(
            text("""
            SELECT month, SUM(feedback_count) AS feedback_count, SUM(rating_sum) AS rating_sum
            FROM analytics_bi.organization_monthly WHERE run_id = :run GROUP BY month ORDER BY month
        """),
            {"run": run_id},
        )
        priorities = await load_problem_priorities(session, run_id)
        return AnalyticalReport(
            **dict(metadata),
            created_at=datetime.now(UTC),
            organizations=tuple(ReportOrganization.model_validate(dict(row)) for row in organizations.mappings()),
            categories=tuple(ReportCategory.model_validate(dict(row)) for row in categories.mappings()),
            months=tuple(ReportMonth.model_validate(dict(row)) for row in months.mappings()),
            recommendations=tuple(
                ReportRecommendation(
                    organization_id=row.signal.organization_id,
                    category_name_ru=row.signal.category_name_ru,
                    score=row.score,
                    requires_review=row.status == "review_required",
                    action_ru=row.recommendation_ru,
                    explanation_ru=row.explanation_ru,
                )
                for row in priorities
            ),
        )
