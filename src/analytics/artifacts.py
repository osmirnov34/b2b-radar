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


class FeedbackReconciliationReport(BaseModel):
    """Privacy-safe comparison of one artifact with its PostgreSQL snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    validated_at: datetime
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_id: int | None = Field(default=None, ge=1)
    expected_records: int = Field(ge=0)
    database_records: int = Field(ge=0)
    matching_records: int = Field(ge=0)
    missing_records: int = Field(ge=0)
    unexpected_records: int = Field(ge=0)
    changed_records: int = Field(ge=0)
    expected_organizations: int = Field(ge=0)
    database_organizations: int = Field(ge=0)
    expected_rating_distribution: dict[int, int]
    database_rating_distribution: dict[int, int]
    expected_published_on_min: date | None = None
    expected_published_on_max: date | None = None
    database_published_on_min: date | None = None
    database_published_on_max: date | None = None
    errors: tuple[str, ...] = ()

    @property
    def is_valid(self) -> bool:
        """Whether database content and metadata match the prepared artifact exactly."""
        return not self.errors

    @model_validator(mode="after")
    def validate_reconciliation_counts(self) -> FeedbackReconciliationReport:
        """Keep the aggregate report internally auditable without exposing rows."""
        if self.matching_records + self.missing_records + self.changed_records != self.expected_records:
            msg = "artifact reconciliation counts do not add up"
            raise ValueError(msg)
        if self.matching_records + self.unexpected_records + self.changed_records != self.database_records:
            msg = "database reconciliation counts do not add up"
            raise ValueError(msg)
        if sum(self.expected_rating_distribution.values()) != self.expected_records:
            msg = "expected rating distribution does not add up"
            raise ValueError(msg)
        if sum(self.database_rating_distribution.values()) != self.database_records:
            msg = "database rating distribution does not add up"
            raise ValueError(msg)
        return self
