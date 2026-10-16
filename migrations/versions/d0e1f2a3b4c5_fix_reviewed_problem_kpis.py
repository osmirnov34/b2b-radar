"""Exclude rejected assessments from problem KPI signals.

Revision ID: d0e1f2a3b4c5
Revises: c9d0e1f2a3b4
"""

from alembic import op

revision = "d0e1f2a3b4c5"
down_revision = "c9d0e1f2a3b4"
branch_labels = None
depends_on = None


def _view_sql(monthly: bool, corrected: bool) -> str:
    name = "problem_monthly_kpis" if monthly else "problem_kpis"
    month = "date_trunc('month', f.published_on)::date"
    time_select = f"{month} AS month," if monthly else ""
    time_group = f", {month}" if monthly else ""
    time_join = f"AND totals.month = {month}" if monthly else ""
    total_month = ", date_trunc('month', published_on)::date AS month" if monthly else ""
    total_group = ", date_trunc('month', published_on)::date" if monthly else ""
    eligible = " AND pa.review_status <> 'rejected'" if corrected else ""
    counts = ",\n".join(
        f"COUNT(DISTINCT pa.feedback_id) FILTER (WHERE pa.stance = '{stance}'{eligible})::bigint AS {stance}_count"
        for stance in ("complaint", "mixed", "praise")
    )
    shares = ",\n".join(
        f"COUNT(DISTINCT pa.feedback_id) FILTER (WHERE pa.stance = '{stance}'{eligible})::double precision "
        f"/ NULLIF(totals.feedback_count, 0) AS {stance}_share"
        for stance in (("complaint",) if monthly else ("complaint", "mixed", "praise"))
    )
    category_select = "pa.category_id," if monthly else (
        "pc.category_id, pc.name_ru AS category_name_ru, pc.impact_hypotheses,"
    )
    category_join = "" if monthly else (
        "JOIN analytics.problem_categories pc ON pc.run_id = pa.run_id AND pc.category_id = pa.category_id"
    )
    category_group = "pa.category_id" if monthly else "pc.category_id, pc.name_ru, pc.impact_hypotheses"
    assessed = "" if monthly else (
        f"COUNT(DISTINCT pa.feedback_id) FILTER (WHERE true{eligible})::bigint AS assessed_feedback_count,"
    )
    quality = "" if monthly else (
        f"AVG(pa.category_probability) FILTER (WHERE true{eligible})::double precision AS mean_category_probability,"
        "COUNT(*) FILTER (WHERE pa.review_status = 'unreviewed')::bigint AS unreviewed_count,"
        "COUNT(*) FILTER (WHERE pa.review_status = 'accepted')::bigint AS accepted_count,"
        "COUNT(*) FILTER (WHERE pa.review_status = 'rejected')::bigint AS rejected_count,"
    )
    review_columns = ""
    if corrected:
        review_columns = "," + ",".join(
            f"COUNT(*) FILTER (WHERE pa.stance = 'complaint' AND pa.review_status = '{status}')::bigint "
            f"AS {status}_complaint_count, "
            f"COUNT(*) FILTER (WHERE pa.stance = 'complaint' AND pa.review_status = '{status}')::double precision "
            f"/ NULLIF(totals.feedback_count, 0) AS {status}_complaint_share"
            for status in ("accepted", "unreviewed")
        )
    return f"""
        CREATE OR REPLACE VIEW analytics.{name} AS
        WITH totals AS (
            SELECT dataset_id, organization_id{total_month}, COUNT(*)::bigint AS feedback_count
            FROM analytics.feedback GROUP BY dataset_id, organization_id{total_group}
        )
        SELECT pa.run_id, pa.dataset_id, f.organization_id, {time_select} {category_select}
               totals.feedback_count, {assessed} {counts}, {shares}, {quality}
               totals.feedback_count >= 30 AS sample_sufficient {review_columns}
        FROM analytics.problem_assessments pa
        JOIN analytics.feedback f ON f.dataset_id = pa.dataset_id AND f.feedback_id = pa.feedback_id
        {category_join}
        JOIN totals ON totals.dataset_id = f.dataset_id AND totals.organization_id = f.organization_id {time_join}
        GROUP BY pa.run_id, pa.dataset_id, f.organization_id{time_group}, {category_group}, totals.feedback_count
    """


def upgrade() -> None:
    """Preserve view columns and append review-specific complaint indicators."""
    op.execute(_view_sql(monthly=False, corrected=True))
    op.execute(_view_sql(monthly=True, corrected=True))


def downgrade() -> None:
    """Restore the preceding view contracts including their original aggregation."""
    for monthly in (True, False):
        name = "problem_monthly_kpis" if monthly else "problem_kpis"
        op.execute(f"DROP VIEW analytics.{name}")
        op.execute(_view_sql(monthly=monthly, corrected=False))
