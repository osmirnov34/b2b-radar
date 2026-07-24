"""add filter settings table

Revision ID: a1b2c3d4e5f6
Revises: fe88a682b292
Create Date: 2026-07-24 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: str | Sequence[str] | None = "fe88a682b292"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "filter_settings",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("source_min_views", sa.Integer(), server_default="0", nullable=False),
        sa.Column("source_min_likes", sa.Integer(), server_default="0", nullable=False),
        sa.Column("source_min_comments", sa.Integer(), server_default="0", nullable=False),
        sa.Column("source_min_duration_seconds", sa.Integer(), server_default="0", nullable=False),
        sa.Column("source_max_age_days", sa.Integer(), server_default="0", nullable=False),
        sa.Column("document_min_likes", sa.Integer(), server_default="0", nullable=False),
        sa.Column("document_min_length", sa.Integer(), server_default="0", nullable=False),
        sa.Column("document_min_replies", sa.Integer(), server_default="0", nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    # Seed the single settings row with defaults so the UI always has a row to edit.
    op.execute("INSERT INTO filter_settings (id) VALUES (1)")


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("filter_settings")
