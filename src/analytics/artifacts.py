"""Persisted contracts for prepared analytical datasets."""

from __future__ import annotations

from datetime import date, datetime  # noqa: TC003 -- Pydantic resolves persisted field types at runtime.

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.analytics.contracts import FeedbackSource  # noqa: TC001 -- Pydantic resolves this enum at runtime.


class FeedbackPreparationManifest(BaseModel):
    """Reproducibility and alignment metadata for one canonical JSONL artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    source: FeedbackSource
    source_path: str
    source_size_bytes: int = Field(ge=0)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_path: str
    output_size_bytes: int = Field(ge=0)
    output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prepared_at: datetime
    records: int = Field(ge=0)
    unique_organizations: int = Field(ge=0)
    rating_distribution: dict[int, int]
    published_on_min: date | None = None
    published_on_max: date | None = None

    @model_validator(mode="after")
    def validate_aggregates(self) -> FeedbackPreparationManifest:
        """Prevent publishing internally inconsistent artifact metadata."""
        if sum(self.rating_distribution.values()) != self.records:
            msg = "rating distribution must add up to records"
            raise ValueError(msg)
        if self.unique_organizations > self.records:
            msg = "unique organizations cannot exceed records"
            raise ValueError(msg)
        if (self.published_on_min is None) != (self.published_on_max is None):
            msg = "publication range must contain both boundaries or neither"
            raise ValueError(msg)
        if (
            self.published_on_min is not None
            and self.published_on_max is not None
            and self.published_on_min > self.published_on_max
        ):
            msg = "publication range is reversed"
            raise ValueError(msg)
        return self
