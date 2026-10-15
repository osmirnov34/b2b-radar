from pathlib import Path

from sqlalchemy import CheckConstraint, ForeignKeyConstraint, UniqueConstraint

from src.infrastructure.analytics_db import ANALYTICS_SCHEMA, AnalyticsBase
from src.infrastructure.db.models import Base

PROJECT_ROOT = Path(__file__).parents[1]


def test_analytics_metadata_is_isolated_from_operational_tables() -> None:
    assert set(AnalyticsBase.metadata.tables) == {
        "analytics.datasets",
        "analytics.feedback",
        "analytics.organizations",
    }
    assert not set(AnalyticsBase.metadata.tables).intersection(Base.metadata.tables)
    assert {table.schema for table in AnalyticsBase.metadata.tables.values()} == {ANALYTICS_SCHEMA}


def test_feedback_fact_has_private_snapshot_keys_and_no_author() -> None:
    table = AnalyticsBase.metadata.tables["analytics.feedback"]

    assert set(table.primary_key.columns.keys()) == {"dataset_id", "feedback_id"}
    assert "text" in table.columns
    assert "author" not in table.columns
    assert {index.name for index in table.indexes} == {
        "ix_analytics_feedback_dataset_organization",
        "ix_analytics_feedback_dataset_published",
        "ix_analytics_feedback_dataset_rating",
    }
    assert any(
        isinstance(constraint, CheckConstraint) and constraint.name == "ck_analytics_feedback_rating"
        for constraint in table.constraints
    )
    assert any(
        isinstance(constraint, UniqueConstraint) and constraint.name == "uq_analytics_feedback_source_record"
        for constraint in table.constraints
    )


def test_analytics_foreign_keys_delete_complete_snapshots() -> None:
    organization = AnalyticsBase.metadata.tables["analytics.organizations"]
    feedback = AnalyticsBase.metadata.tables["analytics.feedback"]

    organization_fk = next(
        constraint for constraint in organization.constraints if isinstance(constraint, ForeignKeyConstraint)
    )
    feedback_fk = next(
        constraint for constraint in feedback.constraints if isinstance(constraint, ForeignKeyConstraint)
    )

    assert organization_fk.ondelete == "CASCADE"
    assert feedback_fk.ondelete == "CASCADE"
    assert {element.target_fullname for element in feedback_fk.elements} == {
        "analytics.organizations.dataset_id",
        "analytics.organizations.organization_id",
    }


def test_analytics_migration_extends_current_head_and_is_reversible() -> None:
    migration = (
        PROJECT_ROOT / "migrations/versions/a7c8d9e0f1a2_add_analytics_feedback_schema.py"
    ).read_text(encoding="utf-8")

    assert 'down_revision: str | Sequence[str] | None = "f6a7b8c9d0e1"' in migration
    assert "CreateSchema(SCHEMA)" in migration
    assert 'op.drop_table("feedback", schema=SCHEMA)' in migration
    assert "DropSchema(SCHEMA)" in migration
