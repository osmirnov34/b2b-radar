"""Aggregate-only contracts for the Russian management report."""

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _ReportModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ReportOrganization(_ReportModel):
    organization_id: str
    organization_name: str
    feedback_count: int = Field(gt=0)
    rating_sum: float = Field(ge=0)
    low_rating_count: float = Field(ge=0)


class ReportCategory(_ReportModel):
    category_id: str
    category_name_ru: str
    feedback_count: int = Field(gt=0)
    complaint_count: int = Field(ge=0)
    mixed_count: int = Field(ge=0)
    praise_count: int = Field(ge=0)
    accepted_complaint_count: int = Field(ge=0)
    unreviewed_complaint_count: int = Field(ge=0)
    rejected_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_counts(self) -> "ReportCategory":
        if self.complaint_count != self.accepted_complaint_count + self.unreviewed_complaint_count:
            msg = "report complaint counters are inconsistent"
            raise ValueError(msg)
        if self.complaint_count + self.mixed_count + self.praise_count > self.feedback_count:
            msg = "category signals exceed source feedback"
            raise ValueError(msg)
        return self


class ReportMonth(_ReportModel):
    month: date
    feedback_count: int = Field(gt=0)
    rating_sum: float = Field(ge=0)


class ReportRecommendation(_ReportModel):
    organization_id: str
    category_name_ru: str
    score: float = Field(ge=0, le=100)
    requires_review: bool
    action_ru: str
    explanation_ru: str


class AnalyticalReport(_ReportModel):
    """Exact run binding and auditable numbers without individual feedback data."""

    schema_version: int = 1
    run_id: int = Field(gt=0)
    dataset_id: int = Field(gt=0)
    dataset_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    assessments_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    taxonomy_id: str
    taxonomy_version: str
    ruleset_version: str
    source: str
    run_generated_at: datetime
    created_at: datetime
    feedback_count: int = Field(gt=0)
    published_on_min: date
    published_on_max: date
    organizations: tuple[ReportOrganization, ...]
    categories: tuple[ReportCategory, ...]
    months: tuple[ReportMonth, ...]
    recommendations: tuple[ReportRecommendation, ...]

    @model_validator(mode="after")
    def reconcile_aggregates(self) -> "AnalyticalReport":
        if sum(row.feedback_count for row in self.organizations) != self.feedback_count:
            msg = "organization counts do not reconcile with the dataset"
            raise ValueError(msg)
        if sum(row.feedback_count for row in self.months) != self.feedback_count:
            msg = "monthly counts do not reconcile with the dataset"
            raise ValueError(msg)
        if not self.categories or any(row.feedback_count != self.feedback_count for row in self.categories):
            msg = "category denominators do not cover the complete dataset"
            raise ValueError(msg)
        return self
