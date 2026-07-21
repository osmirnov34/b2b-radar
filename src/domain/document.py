from datetime import datetime
from typing import Any

import pydantic
from pydantic import Field


class Document(pydantic.BaseModel):
    """Document: YouTube video, Telegram message, or news from the site."""

    id: int | None = None
    source_id: int
    external_id: str
    text: str
    created_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)
    extracted_at: datetime | None = None
