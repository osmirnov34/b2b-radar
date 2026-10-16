"""Canonical records shared by analytics storage, metrics, and reporting."""

from __future__ import annotations

from datetime import date  # noqa: TC003 -- Pydantic resolves this runtime field type.
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


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


class FeedbackTopicAssignment(BaseModel):
    """Privacy-safe analytical projection of one feedback record and its ML topic."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    feedback_id: str = Field(min_length=1)
    source: FeedbackSource
    source_record_id: str = Field(min_length=1)
    rating: int = Field(ge=1, le=5, strict=True)
    published_on: date
    organization_id: str = Field(min_length=1)
    organization_name: str = Field(min_length=1)
    organization_url: str = Field(min_length=1)
    search_query: str = ""
    corpus_id: str = Field(pattern=r"^corpus:[0-9a-f]{64}$")
    topic_id: int | None = Field(default=None, ge=0)
    topic_name: str | None = None
    confidence: float = Field(ge=0, le=1)
    reassigned: bool = False

    @model_validator(mode="after")
    def validate_topic(self) -> FeedbackTopicAssignment:
        """Keep outlier and assigned-topic representations unambiguous."""
        if (self.topic_id is None) != (self.topic_name is None):
            msg = "topic_id and topic_name must both be present or absent"
            raise ValueError(msg)
        if self.reassigned and self.topic_id is None:
            msg = "an outlier cannot be marked as reassigned"
            raise ValueError(msg)
        return self
