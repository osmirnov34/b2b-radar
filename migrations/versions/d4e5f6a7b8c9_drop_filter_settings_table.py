"""drop filter settings table

Revision ID: d4e5f6a7b8c9
Revises: c3d4e5f6a7b8
Create Date: 2026-07-25 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d4e5f6a7b8c9"
down_revision: str | Sequence[str] | None = "c3d4e5f6a7b8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Comment-level numeric thresholds are gone; the language gate is now the only ingestion filter.
    op.drop_table("filter_settings")


def downgrade() -> None:
    """Downgrade schema."""
    op.create_table(
        "filter_settings",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("document_min_likes", sa.Integer(), server_default="0", nullable=False),
        sa.Column("document_min_length", sa.Integer(), server_default="0", nullable=False),
        sa.Column("document_min_replies", sa.Integer(), server_default="0", nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.execute("INSERT INTO filter_settings (id) VALUES (1)")
