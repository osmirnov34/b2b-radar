"""add analytics feedback schema

Revision ID: a7c8d9e0f1a2
Revises: f6a7b8c9d0e1
Create Date: 2026-10-15 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a7c8d9e0f1a2"
down_revision: str | Sequence[str] | None = "f6a7b8c9d0e1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "analytics"


def upgrade() -> None:
    """Create the isolated analytical snapshot schema."""
    op.execute(sa.schema.CreateSchema(SCHEMA))
    op.create_table(
        "datasets",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("source", sa.String(length=50), nullable=False),
        sa.Column("manifest_schema_version", sa.SmallInteger(), nullable=False),
        sa.Column("source_sha256", sa.String(length=64), nullable=False),
        sa.Column("artifact_sha256", sa.String(length=64), nullable=False),
        sa.Column("prepared_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("imported_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("record_count", sa.BigInteger(), nullable=False),
        sa.Column("organization_count", sa.BigInteger(), nullable=False),
        sa.Column("published_on_min", sa.Date(), nullable=True),
        sa.Column("published_on_max", sa.Date(), nullable=True),
        sa.CheckConstraint("organization_count >= 0", name="ck_analytics_datasets_organization_count"),
        sa.CheckConstraint(
            "organization_count <= record_count",
            name="ck_analytics_datasets_organization_within_records",
        ),
        sa.CheckConstraint(
            "published_on_min IS NULL OR published_on_max IS NULL OR published_on_min <= published_on_max",
            name="ck_analytics_datasets_publication_range",
        ),
        sa.CheckConstraint("record_count >= 0", name="ck_analytics_datasets_record_count"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("artifact_sha256", name="uq_analytics_datasets_artifact_sha256"),
        schema=SCHEMA,
    )
    op.create_table(
        "organizations",
        sa.Column("dataset_id", sa.BigInteger(), nullable=False),
        sa.Column("organization_id", sa.String(length=255), nullable=False),
        sa.Column("organization_name", sa.Text(), nullable=False),
        sa.Column("organization_url", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ["dataset_id"],
            [f"{SCHEMA}.datasets.id"],
            name="fk_analytics_organizations_dataset",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("dataset_id", "organization_id"),
        schema=SCHEMA,
    )
    op.create_table(
        "feedback",
        sa.Column("dataset_id", sa.BigInteger(), nullable=False),
        sa.Column("feedback_id", sa.String(length=320), nullable=False),
        sa.Column("source_record_id", sa.String(length=255), nullable=False),
        sa.Column("organization_id", sa.String(length=255), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("rating", sa.SmallInteger(), nullable=False),
        sa.Column("published_on", sa.Date(), nullable=False),
        sa.Column("search_query", sa.Text(), server_default="", nullable=False),
        sa.CheckConstraint("rating BETWEEN 1 AND 5", name="ck_analytics_feedback_rating"),
        sa.ForeignKeyConstraint(
            ["dataset_id", "organization_id"],
            [f"{SCHEMA}.organizations.dataset_id", f"{SCHEMA}.organizations.organization_id"],
            name="fk_analytics_feedback_organization",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("dataset_id", "feedback_id"),
        sa.UniqueConstraint("dataset_id", "source_record_id", name="uq_analytics_feedback_source_record"),
        schema=SCHEMA,
    )
    op.create_index(
        "ix_analytics_feedback_dataset_published",
        "feedback",
        ["dataset_id", "published_on"],
        schema=SCHEMA,
    )
    op.create_index(
        "ix_analytics_feedback_dataset_organization",
        "feedback",
        ["dataset_id", "organization_id"],
        schema=SCHEMA,
    )
    op.create_index(
        "ix_analytics_feedback_dataset_rating",
        "feedback",
        ["dataset_id", "rating"],
        schema=SCHEMA,
    )


def downgrade() -> None:
    """Remove only the analytical schema introduced by this revision."""
    op.drop_index("ix_analytics_feedback_dataset_rating", table_name="feedback", schema=SCHEMA)
    op.drop_index("ix_analytics_feedback_dataset_organization", table_name="feedback", schema=SCHEMA)
    op.drop_index("ix_analytics_feedback_dataset_published", table_name="feedback", schema=SCHEMA)
    op.drop_table("feedback", schema=SCHEMA)
    op.drop_table("organizations", schema=SCHEMA)
    op.drop_table("datasets", schema=SCHEMA)
    op.execute(sa.schema.DropSchema(SCHEMA))
