"""Read-only PostgreSQL boundary for explainable analytical priorities."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import text

from src.analytics.priorities import ProblemPriority, ProblemPriorityInput, rank_problem_priorities

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


async def load_problem_priorities(session: AsyncSession, run_id: int) -> list[ProblemPriority]:
    """Load one explicit run from corrected KPI views and rank its complaint signals."""
    if run_id < 1:
        msg = "run_id must be positive"
        raise ValueError(msg)
    rows = await session.execute(
        text("""
            WITH periods AS (
                SELECT date_trunc('month', generated_at AT TIME ZONE 'UTC')::date AS end_month
                FROM analytics.problem_runs WHERE id = :run_id
            ), monthly AS (
                SELECT k.organization_id, k.category_id,
                       date_trunc('month', f.published_on)::date AS month,
                       COUNT(*) AS feedback_count,
                       COUNT(pa.feedback_id)::double precision / COUNT(*) AS complaint_share
                FROM analytics.problem_kpis k
                CROSS JOIN periods p
                JOIN analytics.feedback f
                  ON f.dataset_id = k.dataset_id AND f.organization_id = k.organization_id
                LEFT JOIN analytics.problem_assessments pa
                  ON pa.dataset_id = f.dataset_id AND pa.feedback_id = f.feedback_id
                 AND pa.run_id = k.run_id AND pa.category_id = k.category_id
                 AND pa.stance = 'complaint' AND pa.review_status <> 'rejected'
                WHERE k.run_id = :run_id AND f.published_on >= p.end_month - INTERVAL '2 months'
                  AND f.published_on < p.end_month
                GROUP BY k.organization_id, k.category_id, date_trunc('month', f.published_on)::date
            )
            SELECT k.*,
                   CASE WHEN prev.feedback_count IS NOT NULL AND curr.feedback_count IS NOT NULL
                        THEN prev.complaint_share END AS previous_complaint_share,
                   CASE WHEN prev.feedback_count IS NOT NULL AND curr.feedback_count IS NOT NULL
                        THEN curr.complaint_share END AS current_complaint_share,
                   COALESCE(prev.feedback_count, 0) AS previous_feedback_count,
                   COALESCE(curr.feedback_count, 0) AS current_feedback_count
            FROM analytics.problem_kpis k CROSS JOIN periods p
            LEFT JOIN monthly prev ON prev.organization_id = k.organization_id AND prev.category_id = k.category_id
              AND prev.month = (p.end_month - INTERVAL '2 months')::date
            LEFT JOIN monthly curr ON curr.organization_id = k.organization_id AND curr.category_id = k.category_id
              AND curr.month = (p.end_month - INTERVAL '1 month')::date
            WHERE k.run_id = :run_id
        """),
        {"run_id": run_id},
    )
    signals = [dict(row) for row in rows.mappings()]
    evidence_rows = await session.execute(
        text("""
            SELECT organization_id, category_id, feedback_id FROM (
                SELECT f.organization_id, pa.category_id, pa.feedback_id,
                       row_number() OVER (
                         PARTITION BY f.organization_id, pa.category_id
                         ORDER BY (pa.review_status = 'accepted') DESC, pa.category_probability DESC, pa.feedback_id
                       ) AS position
                FROM analytics.problem_assessments pa
                JOIN analytics.feedback f ON f.dataset_id = pa.dataset_id AND f.feedback_id = pa.feedback_id
                WHERE pa.run_id = :run_id AND pa.stance = 'complaint' AND pa.review_status <> 'rejected'
            ) ranked WHERE position <= 3
        """),
        {"run_id": run_id},
    )
    evidence: dict[tuple[str, str], list[str]] = {}
    for organization_id, category_id, feedback_id in evidence_rows.tuples():
        evidence.setdefault((organization_id, category_id), []).append(feedback_id)
    for signal in signals:
        signal["evidence_feedback_ids"] = evidence.get((signal["organization_id"], signal["category_id"]), [])
    return rank_problem_priorities([ProblemPriorityInput.model_validate(signal) for signal in signals])
