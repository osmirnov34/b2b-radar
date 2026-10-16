"""add analytics KPI marts

Revision ID: c9d0e1f2a3b4
Revises: b8c9d0e1f2a3
Create Date: 2026-10-16 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c9d0e1f2a3b4"
down_revision: str | Sequence[str] | None = "b8c9d0e1f2a3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "analytics"
MINIMUM_SAMPLE = 30


def upgrade() -> None:
    """Add taxonomy snapshots and aggregate-only BI views."""
    op.create_table(
        "problem_categories",
        sa.Column("run_id", sa.BigInteger(), nullable=False),
        sa.Column("category_id", sa.String(length=64), nullable=False),
        sa.Column("name_ru", sa.Text(), nullable=False),
        sa.Column("definition_ru", sa.Text(), nullable=False),
        sa.Column("impact_hypotheses", sa.Text(), nullable=False),
        sa.Column("active", sa.Boolean(), nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id"],
            [f"{SCHEMA}.problem_runs.id"],
            name="fk_analytics_problem_categories_run",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("run_id", "category_id"),
        schema=SCHEMA,
    )
    op.execute(
        f"""
        CREATE VIEW {SCHEMA}.organization_kpis AS
        SELECT
            f.dataset_id,
            f.organization_id,
            o.organization_name,
            o.organization_url,
            COUNT(*)::bigint AS feedback_count,
            AVG(f.rating)::double precision AS average_rating,
            COUNT(*) FILTER (WHERE f.rating = 1)::bigint AS rating_1_count,
            COUNT(*) FILTER (WHERE f.rating = 2)::bigint AS rating_2_count,
            COUNT(*) FILTER (WHERE f.rating = 3)::bigint AS rating_3_count,
            COUNT(*) FILTER (WHERE f.rating = 4)::bigint AS rating_4_count,
            COUNT(*) FILTER (WHERE f.rating = 5)::bigint AS rating_5_count,
            COUNT(*) FILTER (WHERE f.rating <= 3)::bigint AS low_rating_count,
            COUNT(*) FILTER (WHERE f.rating <= 3)::double precision / NULLIF(COUNT(*), 0) AS low_rating_share,
            COUNT(*) FILTER (WHERE f.rating >= 4)::bigint AS high_rating_count,
            COUNT(*) FILTER (WHERE f.rating >= 4)::double precision / NULLIF(COUNT(*), 0) AS high_rating_share,
            MIN(f.published_on) AS first_feedback_on,
            MAX(f.published_on) AS last_feedback_on,
            COUNT(*) >= {MINIMUM_SAMPLE} AS sample_sufficient
        FROM {SCHEMA}.feedback AS f
        JOIN {SCHEMA}.organizations AS o
          ON o.dataset_id = f.dataset_id AND o.organization_id = f.organization_id
        GROUP BY f.dataset_id, f.organization_id, o.organization_name, o.organization_url
        """,
    )
    op.execute(
        f"""
        CREATE VIEW {SCHEMA}.problem_kpis AS
        WITH organization_totals AS (
            SELECT dataset_id, organization_id, COUNT(*)::bigint AS feedback_count
            FROM {SCHEMA}.feedback
            GROUP BY dataset_id, organization_id
        )
        SELECT
            pa.run_id,
            pa.dataset_id,
            f.organization_id,
            pc.category_id,
            pc.name_ru AS category_name_ru,
            pc.impact_hypotheses,
            totals.feedback_count,
            COUNT(DISTINCT pa.feedback_id)::bigint AS assessed_feedback_count,
            COUNT(DISTINCT pa.feedback_id) FILTER (WHERE pa.stance = 'complaint')::bigint AS complaint_count,
            COUNT(DISTINCT pa.feedback_id) FILTER (WHERE pa.stance = 'mixed')::bigint AS mixed_count,
            COUNT(DISTINCT pa.feedback_id) FILTER (WHERE pa.stance = 'praise')::bigint AS praise_count,
            COUNT(DISTINCT pa.feedback_id) FILTER (WHERE pa.stance = 'complaint')::double precision
                / NULLIF(totals.feedback_count, 0) AS complaint_share,
            COUNT(DISTINCT pa.feedback_id) FILTER (WHERE pa.stance = 'mixed')::double precision
                / NULLIF(totals.feedback_count, 0) AS mixed_share,
            COUNT(DISTINCT pa.feedback_id) FILTER (WHERE pa.stance = 'praise')::double precision
                / NULLIF(totals.feedback_count, 0) AS praise_share,
            AVG(pa.category_probability)::double precision AS mean_category_probability,
            COUNT(*) FILTER (WHERE pa.review_status = 'unreviewed')::bigint AS unreviewed_count,
            COUNT(*) FILTER (WHERE pa.review_status = 'accepted')::bigint AS accepted_count,
            COUNT(*) FILTER (WHERE pa.review_status = 'rejected')::bigint AS rejected_count,
            totals.feedback_count >= {MINIMUM_SAMPLE} AS sample_sufficient
        FROM {SCHEMA}.problem_assessments AS pa
        JOIN {SCHEMA}.feedback AS f
          ON f.dataset_id = pa.dataset_id AND f.feedback_id = pa.feedback_id
        JOIN {SCHEMA}.problem_categories AS pc
          ON pc.run_id = pa.run_id AND pc.category_id = pa.category_id
        JOIN organization_totals AS totals
          ON totals.dataset_id = pa.dataset_id AND totals.organization_id = f.organization_id
        GROUP BY pa.run_id, pa.dataset_id, f.organization_id, pc.category_id, pc.name_ru,
                 pc.impact_hypotheses, totals.feedback_count
        """,
    )
    op.execute(
        f"""
        CREATE VIEW {SCHEMA}.organization_monthly_kpis AS
        SELECT
            f.dataset_id,
            f.organization_id,
            date_trunc('month', f.published_on)::date AS month,
            COUNT(*)::bigint AS feedback_count,
            AVG(f.rating)::double precision AS average_rating,
            COUNT(*) FILTER (WHERE f.rating <= 3)::double precision / NULLIF(COUNT(*), 0) AS low_rating_share,
            COUNT(*) >= {MINIMUM_SAMPLE} AS sample_sufficient
        FROM {SCHEMA}.feedback AS f
        GROUP BY f.dataset_id, f.organization_id, date_trunc('month', f.published_on)::date
        """,
    )
    op.execute(
        f"""
        CREATE VIEW {SCHEMA}.problem_monthly_kpis AS
        WITH monthly_totals AS (
            SELECT dataset_id, organization_id, date_trunc('month', published_on)::date AS month,
                   COUNT(*)::bigint AS feedback_count
            FROM {SCHEMA}.feedback
            GROUP BY dataset_id, organization_id, date_trunc('month', published_on)::date
        )
        SELECT
            pa.run_id,
            pa.dataset_id,
            f.organization_id,
            date_trunc('month', f.published_on)::date AS month,
            pa.category_id,
            totals.feedback_count,
            COUNT(DISTINCT pa.feedback_id) FILTER (WHERE pa.stance = 'complaint')::bigint AS complaint_count,
            COUNT(DISTINCT pa.feedback_id) FILTER (WHERE pa.stance = 'mixed')::bigint AS mixed_count,
            COUNT(DISTINCT pa.feedback_id) FILTER (WHERE pa.stance = 'praise')::bigint AS praise_count,
            COUNT(DISTINCT pa.feedback_id) FILTER (WHERE pa.stance = 'complaint')::double precision
                / NULLIF(totals.feedback_count, 0) AS complaint_share,
            totals.feedback_count >= {MINIMUM_SAMPLE} AS sample_sufficient
        FROM {SCHEMA}.problem_assessments AS pa
        JOIN {SCHEMA}.feedback AS f
          ON f.dataset_id = pa.dataset_id AND f.feedback_id = pa.feedback_id
        JOIN monthly_totals AS totals
          ON totals.dataset_id = f.dataset_id
         AND totals.organization_id = f.organization_id
         AND totals.month = date_trunc('month', f.published_on)::date
        GROUP BY pa.run_id, pa.dataset_id, f.organization_id, date_trunc('month', f.published_on)::date,
                 pa.category_id, totals.feedback_count
        """,
    )


def downgrade() -> None:
    """Remove aggregate views and taxonomy snapshots."""
    op.execute(f"DROP VIEW {SCHEMA}.problem_monthly_kpis")
    op.execute(f"DROP VIEW {SCHEMA}.organization_monthly_kpis")
    op.execute(f"DROP VIEW {SCHEMA}.problem_kpis")
    op.execute(f"DROP VIEW {SCHEMA}.organization_kpis")
    op.drop_table("problem_categories", schema=SCHEMA)
