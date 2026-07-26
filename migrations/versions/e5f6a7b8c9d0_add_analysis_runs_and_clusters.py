"""add analysis runs and clusters

Revision ID: e5f6a7b8c9d0
Revises: d4e5f6a7b8c9
Create Date: 2026-07-25 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision: str = "e5f6a7b8c9d0"
down_revision: str | Sequence[str] | None = "d4e5f6a7b8c9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "analysis_runs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("label", sa.String(length=255), nullable=False),
        sa.Column("model", sa.String(length=255), nullable=True),
        sa.Column("near_dup_threshold", sa.Float(), nullable=True),
        sa.Column("min_topic_size", sa.Integer(), nullable=True),
        sa.Column("n_clusters", sa.Integer(), nullable=False),
        sa.Column("n_comments", sa.Integer(), nullable=False),
        sa.Column("n_authors", sa.Integer(), nullable=False),
        sa.Column("n_channels", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_analysis_runs_created_at", "analysis_runs", ["created_at"])

    op.create_table(
        "clusters",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("run_id", sa.Integer(), nullable=False),
        sa.Column("topic_id", sa.Integer(), nullable=False),
        sa.Column("n_comments", sa.Integer(), nullable=False),
        sa.Column("n_authors", sa.Integer(), nullable=False),
        sa.Column("n_channels", sa.Integer(), nullable=False),
        sa.Column("keywords", JSONB(), nullable=False),
        sa.Column("comments", JSONB(), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["analysis_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_clusters_run_id", "clusters", ["run_id"])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_clusters_run_id", table_name="clusters")
    op.drop_table("clusters")
    op.drop_index("ix_analysis_runs_created_at", table_name="analysis_runs")
    op.drop_table("analysis_runs")
