from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, String, func
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
    extracted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)


class YoutubeApiKeyModel(Base):
    """Database model representing a YouTube Data API key, managed by an admin."""

    __tablename__ = "youtube_api_keys"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str | None] = mapped_column(String(255))
    key: Mapped[str] = mapped_column(String(255), unique=True)
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
