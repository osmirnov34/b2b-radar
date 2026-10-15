"""Relational analytical snapshot models isolated from the application database models."""

from __future__ import annotations

from datetime import date, datetime  # noqa: TC003 -- SQLAlchemy resolves mapped field types at runtime.

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKeyConstraint,
    Index,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

ANALYTICS_SCHEMA = "analytics"


class AnalyticsBase(AsyncAttrs, DeclarativeBase):
    """Separate metadata boundary for analytical tables."""


class AnalyticsDatasetModel(AnalyticsBase):
    """One immutable prepared dataset imported for analysis and BI."""

    __tablename__ = "datasets"
    __table_args__ = (
        CheckConstraint("record_count >= 0", name="ck_analytics_datasets_record_count"),
        CheckConstraint("organization_count >= 0", name="ck_analytics_datasets_organization_count"),
        CheckConstraint(
            "organization_count <= record_count",
            name="ck_analytics_datasets_organization_within_records",
        ),
        CheckConstraint(
            "published_on_min IS NULL OR published_on_max IS NULL OR published_on_min <= published_on_max",
            name="ck_analytics_datasets_publication_range",
        ),
        UniqueConstraint("artifact_sha256", name="uq_analytics_datasets_artifact_sha256"),
        {"schema": ANALYTICS_SCHEMA},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(50))
    manifest_schema_version: Mapped[int] = mapped_column(SmallInteger)
    source_sha256: Mapped[str] = mapped_column(String(64))
    artifact_sha256: Mapped[str] = mapped_column(String(64))
    prepared_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    imported_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    record_count: Mapped[int] = mapped_column(BigInteger)
    organization_count: Mapped[int] = mapped_column(BigInteger)
    published_on_min: Mapped[date | None] = mapped_column(Date, default=None)
    published_on_max: Mapped[date | None] = mapped_column(Date, default=None)


class AnalyticsOrganizationModel(AnalyticsBase):
    """Organization dimension scoped to one immutable dataset snapshot."""

    __tablename__ = "organizations"
    __table_args__ = (
        ForeignKeyConstraint(
            ["dataset_id"],
            [f"{ANALYTICS_SCHEMA}.datasets.id"],
            ondelete="CASCADE",
            name="fk_analytics_organizations_dataset",
        ),
        {"schema": ANALYTICS_SCHEMA},
    )

    dataset_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    organization_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    organization_name: Mapped[str] = mapped_column(Text)
    organization_url: Mapped[str] = mapped_column(Text)


class AnalyticsFeedbackModel(AnalyticsBase):
    """Private feedback fact table with explicit snapshot and organization provenance."""

    __tablename__ = "feedback"
    __table_args__ = (
        ForeignKeyConstraint(
            ["dataset_id", "organization_id"],
            [
                f"{ANALYTICS_SCHEMA}.organizations.dataset_id",
                f"{ANALYTICS_SCHEMA}.organizations.organization_id",
            ],
            ondelete="CASCADE",
            name="fk_analytics_feedback_organization",
        ),
        CheckConstraint("rating BETWEEN 1 AND 5", name="ck_analytics_feedback_rating"),
        UniqueConstraint("dataset_id", "source_record_id", name="uq_analytics_feedback_source_record"),
        Index("ix_analytics_feedback_dataset_published", "dataset_id", "published_on"),
        Index("ix_analytics_feedback_dataset_organization", "dataset_id", "organization_id"),
        Index("ix_analytics_feedback_dataset_rating", "dataset_id", "rating"),
        {"schema": ANALYTICS_SCHEMA},
    )

    dataset_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    feedback_id: Mapped[str] = mapped_column(String(320), primary_key=True)
    source_record_id: Mapped[str] = mapped_column(String(255))
    organization_id: Mapped[str] = mapped_column(String(255))
    text: Mapped[str] = mapped_column(Text)
    rating: Mapped[int] = mapped_column(SmallInteger)
    published_on: Mapped[date] = mapped_column(Date)
    search_query: Mapped[str] = mapped_column(Text, default="", server_default="")
