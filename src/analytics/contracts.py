"""Canonical records shared by analytics storage, metrics, and reporting."""

from __future__ import annotations

from datetime import date  # noqa: TC003 -- Pydantic resolves this runtime field type.
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class FeedbackSource(StrEnum):
    """Supported origins for a canonical feedback record."""

    YANDEX_MAPS = "yandex_maps"


class FeedbackRecord(BaseModel):
    """One source-neutral customer feedback observation.

    Author display names are intentionally excluded: they are not reliable
    customer identifiers and are unnecessary for the first analytical metrics.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    feedback_id: str = Field(min_length=1)
    source: FeedbackSource
    source_record_id: str = Field(min_length=1)
    text: str
    rating: int = Field(ge=1, le=5, strict=True)
    published_on: date
    organization_id: str = Field(min_length=1)
    organization_name: str = Field(min_length=1)
    organization_url: str = Field(min_length=1)
    search_query: str = ""
