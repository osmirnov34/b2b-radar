"""drop source quality filter columns

Revision ID: c3d4e5f6a7b8
Revises: b2c3d4e5f6a7
Create Date: 2026-07-25 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c3d4e5f6a7b8"
down_revision: str | Sequence[str] | None = "b2c3d4e5f6a7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Video-level thresholds are replaced by the language gate applied in the pipeline;
    # only comment-level quality settings remain user-configurable.
    op.drop_column("filter_settings", "source_min_views")
    op.drop_column("filter_settings", "source_min_likes")
    op.drop_column("filter_settings", "source_min_comments")
    op.drop_column("filter_settings", "source_min_duration_seconds")
    op.drop_column("filter_settings", "source_max_age_days")


def downgrade() -> None:
    """Downgrade schema."""
    op.add_column(
        "filter_settings", sa.Column("source_min_views", sa.Integer(), server_default="0", nullable=False)
    )
    op.add_column(
        "filter_settings", sa.Column("source_min_likes", sa.Integer(), server_default="0", nullable=False)
    )
    op.add_column(
        "filter_settings", sa.Column("source_min_comments", sa.Integer(), server_default="0", nullable=False)
    )
    op.add_column(
        "filter_settings",
        sa.Column("source_min_duration_seconds", sa.Integer(), server_default="0", nullable=False),
    )
    op.add_column(
        "filter_settings", sa.Column("source_max_age_days", sa.Integer(), server_default="0", nullable=False)
    )
