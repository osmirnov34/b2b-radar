from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class SourceType(StrEnum):
    YOUTUBE_VIDEO = "youtube_video"


class Source(BaseModel):
    """Source of b2b ideas: YouTube channel, Telegram chat, or site."""

    type: SourceType
    url: str
    name: str
    metadata: dict[str, Any] = Field(default_factory=dict)
