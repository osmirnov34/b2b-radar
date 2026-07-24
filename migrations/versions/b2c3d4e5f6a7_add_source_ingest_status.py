"""add source ingest status

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-07-24 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b2c3d4e5f6a7"
down_revision: str | Sequence[str] | None = "a1b2c3d4e5f6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "sources",
        sa.Column("ingest_status", sa.String(length=20), server_default="pending", nullable=False),
    )
    op.add_column("sources", sa.Column("ingest_error", sa.String(), nullable=True))
    op.create_index("ix_sources_ingest_status", "sources", ["ingest_status"])

    # Backfill: sources that already have documents are treated as successfully ingested.
    op.execute(
        "UPDATE sources SET ingest_status = 'success' "
        "WHERE id IN (SELECT DISTINCT source_id FROM documents)",
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_sources_ingest_status", table_name="sources")
    op.drop_column("sources", "ingest_error")
    op.drop_column("sources", "ingest_status")
