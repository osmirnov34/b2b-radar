"""add analytics problem assessments

Revision ID: b8c9d0e1f2a3
Revises: a7c8d9e0f1a2
Create Date: 2026-10-16 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b8c9d0e1f2a3"
down_revision: str | Sequence[str] | None = "a7c8d9e0f1a2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "analytics"


def upgrade() -> None:
    """Add immutable problem runs and their category-level assessments."""
    op.create_table(
        "problem_runs",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("dataset_id", sa.BigInteger(), nullable=False),
        sa.Column("artifact_sha256", sa.String(length=64), nullable=False),
        sa.Column("taxonomy_id", sa.String(length=64), nullable=False),
        sa.Column("taxonomy_version", sa.String(length=32), nullable=False),
        sa.Column("ruleset_id", sa.String(length=64), nullable=False),
        sa.Column("ruleset_version", sa.String(length=32), nullable=False),
        sa.Column("topic_assignments_sha256", sa.String(length=64), nullable=True),
        sa.Column("generated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("imported_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("feedback_count", sa.BigInteger(), nullable=False),
        sa.Column("assessed_feedback_count", sa.BigInteger(), nullable=False),
        sa.Column("assessment_count", sa.BigInteger(), nullable=False),
        sa.CheckConstraint("assessment_count >= 0", name="ck_analytics_problem_runs_assessment_count"),
        sa.CheckConstraint("feedback_count >= 0", name="ck_analytics_problem_runs_feedback_count"),
        sa.ForeignKeyConstraint(
            ["dataset_id"],
            [f"{SCHEMA}.datasets.id"],
            name="fk_analytics_problem_runs_dataset",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("artifact_sha256", name="uq_analytics_problem_runs_artifact_sha256"),
        schema=SCHEMA,
    )
    op.create_table(
        "problem_assessments",
        sa.Column("run_id", sa.BigInteger(), nullable=False),
        sa.Column("dataset_id", sa.BigInteger(), nullable=False),
        sa.Column("feedback_id", sa.String(length=320), nullable=False),
        sa.Column("category_id", sa.String(length=64), nullable=False),
        sa.Column("stance", sa.String(length=32), nullable=False),
        sa.Column("category_probability", sa.Float(), nullable=False),
        sa.Column("stance_probability", sa.Float(), nullable=False),
        sa.Column("method", sa.String(length=32), nullable=False),
        sa.Column("evidence_spans", sa.Text(), nullable=False),
        sa.Column("model_name", sa.Text(), nullable=True),
        sa.Column("model_revision", sa.Text(), nullable=True),
        sa.Column("review_status", sa.String(length=32), nullable=False),
        sa.Column("reviewer_id", sa.Text(), nullable=True),
        sa.Column("assessed_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "category_probability BETWEEN 0 AND 1",
            name="ck_analytics_problem_assessments_category_probability",
        ),
        sa.CheckConstraint(
            "stance_probability BETWEEN 0 AND 1",
            name="ck_analytics_problem_assessments_stance_probability",
        ),
        sa.ForeignKeyConstraint(
            ["dataset_id", "feedback_id"],
            [f"{SCHEMA}.feedback.dataset_id", f"{SCHEMA}.feedback.feedback_id"],
            name="fk_analytics_problem_assessments_feedback",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            [f"{SCHEMA}.problem_runs.id"],
            name="fk_analytics_problem_assessments_run",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("run_id", "feedback_id", "category_id"),
        schema=SCHEMA,
    )
    op.create_index(
        "ix_analytics_problem_assessments_dataset_category",
        "problem_assessments",
        ["dataset_id", "category_id"],
        schema=SCHEMA,
    )
    op.create_index(
        "ix_analytics_problem_assessments_dataset_stance",
        "problem_assessments",
        ["dataset_id", "stance"],
        schema=SCHEMA,
    )


def downgrade() -> None:
    """Remove only problem assessment storage."""
    op.drop_index(
        "ix_analytics_problem_assessments_dataset_stance",
        table_name="problem_assessments",
        schema=SCHEMA,
    )
    op.drop_index(
        "ix_analytics_problem_assessments_dataset_category",
        table_name="problem_assessments",
        schema=SCHEMA,
    )
    op.drop_table("problem_assessments", schema=SCHEMA)
    op.drop_table("problem_runs", schema=SCHEMA)
