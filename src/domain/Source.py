from enum import StrEnum
from typing import Any

import pydantic


class SourceType(StrEnum):
    YOUTUBE_VIDEO = "youtube_video"


class Source(pydantic.BaseModel):
    type: SourceType
    url: str
    name: str
    metadata: dict[str, Any] = pydantic.Field(default_factory=dict)
