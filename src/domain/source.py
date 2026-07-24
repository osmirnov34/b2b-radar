from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class SourceType(StrEnum):
    YOUTUBE_VIDEO = "youtube_video"


class IngestStatus(StrEnum):
    """Lifecycle of a source's document (comment) extraction, tracked for observability.

    ``PENDING`` — created, extraction not started yet; ``RUNNING`` — extraction in progress;
    ``SUCCESS`` — extraction finished without error; ``FAILED`` — extraction raised (see ``ingest_error``).
    """

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


class Source(BaseModel):
    """Source of b2b ideas: YouTube channel, Telegram chat, or site."""

    id: int | None = None
    type: SourceType
    url: str
    name: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    ingest_status: IngestStatus = IngestStatus.PENDING
    ingest_error: str | None = None
    extracted_at: datetime | None = None
