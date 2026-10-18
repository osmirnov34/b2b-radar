"""Add aggregate DataLens sources with complete organization/category coverage.

Revision ID: e1f2a3b4c5d6
Revises: d0e1f2a3b4c5
"""

from alembic import op

revision = "e1f2a3b4c5d6"
down_revision = "d0e1f2a3b4c5"
branch_labels = None
depends_on = None


def source_sql(monthly: bool, categories: bool) -> str:
    """Define a fixed aggregate grain; absent category signals become zero observations."""
    base = "organization_monthly_kpis" if monthly else "organization_kpis"
    target = ("category" if categories else "organization") + ("_monthly" if monthly else "_overview")
    month = "o.month," if monthly else ""
    common = f"""
        r.id AS run_id, r.dataset_id, o.organization_id, names.organization_name,
        {month} o.feedback_count, o.sample_sufficient
    """
    if not categories:
        distribution = "" if monthly else ", o.rating_1_count, o.rating_2_count, o.rating_3_count, o.rating_4_count, o.rating_5_count"
        columns = common + f""",
            o.average_rating * o.feedback_count AS rating_sum,
            o.low_rating_share * o.feedback_count AS low_rating_count {distribution}
        """
        joins = ""
    else:
        base_problem = "problem_monthly_kpis" if monthly else "problem_kpis"
        month_join = "AND p.month = o.month" if monthly else ""
        quality = "" if monthly else """,
            COALESCE(p.unreviewed_count, 0) AS unreviewed_count,
            COALESCE(p.accepted_count, 0) AS accepted_count,
            COALESCE(p.rejected_count, 0) AS rejected_count
        """
        columns = common + f""",
            c.category_id, c.name_ru AS category_name_ru, c.active AS category_active,
            COALESCE(p.complaint_count, 0) AS complaint_count,
            COALESCE(p.mixed_count, 0) AS mixed_count,
            COALESCE(p.praise_count, 0) AS praise_count,
            COALESCE(p.accepted_complaint_count, 0) AS accepted_complaint_count,
            COALESCE(p.unreviewed_complaint_count, 0) AS unreviewed_complaint_count
            {quality}
        """
        joins = f"""
            JOIN analytics.problem_categories c ON c.run_id = r.id
            LEFT JOIN analytics.{base_problem} p
              ON p.run_id = r.id AND p.organization_id = o.organization_id
             AND p.category_id = c.category_id {month_join}
        """
    return f"""
        CREATE VIEW analytics_bi.{target} WITH (security_barrier = true) AS
        SELECT {columns}
        FROM analytics.problem_runs r
        JOIN analytics.{base} o ON o.dataset_id = r.dataset_id
        JOIN analytics.organizations names
          ON names.dataset_id = o.dataset_id AND names.organization_id = o.organization_id
        {joins}
    """


def upgrade() -> None:
    """Expose only aggregates in a dedicated schema for a restricted BI account."""
    op.execute("CREATE SCHEMA analytics_bi")
    for monthly in (False, True):
        for categories in (False, True):
            op.execute(source_sql(monthly=monthly, categories=categories))


def downgrade() -> None:
    """Remove only the DataLens sources introduced by this revision."""
    for name in ("category_monthly", "organization_monthly", "category_overview", "organization_overview"):
        op.execute(f"DROP VIEW analytics_bi.{name}")
    op.execute("DROP SCHEMA analytics_bi")
