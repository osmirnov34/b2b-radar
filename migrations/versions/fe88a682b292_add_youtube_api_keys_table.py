"""add youtube api keys table with extracted_at and name

Revision ID: fe88a682b292
Revises: e8132e591740
Create Date: 2026-07-19 13:13:49.896350

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "fe88a682b292"
down_revision: str | Sequence[str] | None = "e8132e591740"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "youtube_api_keys",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("key", sa.String(length=255), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("key"),
    )

    op.add_column(
        "sources",
        sa.Column("extracted_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.create_index(op.f("ix_sources_extracted_at"), "sources", ["extracted_at"], unique=False)

    op.add_column(
        "documents",
        sa.Column("extracted_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.create_index(op.f("ix_documents_extracted_at"), "documents", ["extracted_at"], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f("ix_documents_extracted_at"), table_name="documents")
    op.drop_column("documents", "extracted_at")

    op.drop_index(op.f("ix_sources_extracted_at"), table_name="sources")
    op.drop_column("sources", "extracted_at")

    op.drop_table("youtube_api_keys")
