from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(AsyncAttrs, DeclarativeBase):
    """Base class for all SQLAlchemy models."""


class DocumentModel(Base):
    """Database model representing a document (e.g., a YouTube video comment, a Telegram channel message, or a website comment)."""  # noqa: E501

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    source_id: Mapped[int] = mapped_column(ForeignKey("sources.id", ondelete="CASCADE"), index=True)
    external_id: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    text: Mapped[str]
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    metadata_data: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    extracted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)


class SourceModel(Base):
    """Database model representing a data source (e.g., a YouTube video, a Telegram channel, or a website)."""

    __tablename__ = "sources"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    type: Mapped[str] = mapped_column(String(50))
    url: Mapped[str] = mapped_column(String(2048), unique=True)
    name: Mapped[str] = mapped_column(String(255))
    metadata_data: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    ingest_status: Mapped[str] = mapped_column(String(20), default="pending", server_default="pending", index=True)
    ingest_error: Mapped[str | None] = mapped_column(default=None)
    extracted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)


class AnalysisRunModel(Base):
    """A single uploaded clustering result (one clusters.jsonl), shown in the analysis history."""

    __tablename__ = "analysis_runs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    label: Mapped[str] = mapped_column(String(255))
    model: Mapped[str | None] = mapped_column(String(255), default=None)
    near_dup_threshold: Mapped[float | None] = mapped_column(Float, default=None)
    min_topic_size: Mapped[int | None] = mapped_column(Integer, default=None)
    n_clusters: Mapped[int] = mapped_column(Integer, default=0)
    n_comments: Mapped[int] = mapped_column(Integer, default=0)
    n_authors: Mapped[int] = mapped_column(Integer, default=0)
    n_channels: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)


class ClusterModel(Base):
    """One discovered cluster within a run: keywords + its comments (with per-comment provenance)."""

    __tablename__ = "clusters"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("analysis_runs.id", ondelete="CASCADE"), index=True)
    topic_id: Mapped[int] = mapped_column(Integer)
    n_comments: Mapped[int] = mapped_column(Integer, default=0)
    n_authors: Mapped[int] = mapped_column(Integer, default=0)
    n_channels: Mapped[int] = mapped_column(Integer, default=0)
    keywords: Mapped[list[Any]] = mapped_column(JSONB, default=list)
    comments: Mapped[list[Any]] = mapped_column(JSONB, default=list)


class YoutubeApiKeyModel(Base):
    """Database model representing a YouTube Data API key, managed by an admin."""

    __tablename__ = "youtube_api_keys"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str | None] = mapped_column(String(255))
    key: Mapped[str] = mapped_column(String(255), unique=True)
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
