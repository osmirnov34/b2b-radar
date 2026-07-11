from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class SourceType(StrEnum):
    YOUTUBE_VIDEO = "youtube_video"


class Source(BaseModel):
    type: SourceType
    url: str
    name: str
    metadata: dict[str, Any] = Field(default_factory=dict)
