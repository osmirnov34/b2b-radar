"""Versioned problem taxonomy and evidence-only assessment contracts."""

from __future__ import annotations

import json
from datetime import datetime  # noqa: TC003 -- Pydantic resolves persisted field types at runtime.
from enum import StrEnum
from itertools import pairwise
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

if TYPE_CHECKING:
    from pathlib import Path

_CATEGORY_ID_PATTERN = r"^[a-z][a-z0-9_]{2,63}$"
_VERSION_PATTERN = r"^[1-9][0-9]*\.[0-9]+\.[0-9]+$"


class _ProblemModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class BusinessImpactHypothesis(StrEnum):
    """Mechanisms to investigate, never measured financial effects by themselves."""

    RETENTION_RISK = "retention_risk"
    CONVERSION_RISK = "conversion_risk"
    OPERATIONAL_COST = "operational_cost"
    REPUTATION_RISK = "reputation_risk"


class FeedbackStance(StrEnum):
    """How a feedback record speaks about one specific category."""

    COMPLAINT = "complaint"
    PRAISE = "praise"
    NEUTRAL = "neutral"
    MIXED = "mixed"
    UNCLEAR = "unclear"


class AssessmentMethod(StrEnum):
    """Provenance of a category assessment."""

    MANUAL = "manual"
    RULE = "rule"
    EMBEDDING = "embedding"
    CLUSTER = "cluster"
    LLM = "llm"
    HYBRID = "hybrid"


class ReviewStatus(StrEnum):
    """Human-review state of an automated or manual assessment."""

    UNREVIEWED = "unreviewed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class ProblemCategory(_ProblemModel):
    """One documented and stable category in a domain taxonomy."""

    category_id: str = Field(pattern=_CATEGORY_ID_PATTERN)
    name_ru: str = Field(min_length=1)
    definition_ru: str = Field(min_length=1)
    inclusion_rules: tuple[str, ...] = Field(min_length=1)
    exclusion_rules: tuple[str, ...] = Field(min_length=1)
    impact_hypotheses: tuple[BusinessImpactHypothesis, ...] = Field(min_length=1)
    active: bool = True

    @field_validator("inclusion_rules", "exclusion_rules")
    @classmethod
    def require_non_blank_rules(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value.strip() for value in values):
            msg = "taxonomy rules cannot be blank"
            raise ValueError(msg)
        return values

    @field_validator("impact_hypotheses")
    @classmethod
    def require_unique_impacts(
        cls,
        values: tuple[BusinessImpactHypothesis, ...],
    ) -> tuple[BusinessImpactHypothesis, ...]:
        if len(values) != len(set(values)):
            msg = "impact hypotheses must be unique"
            raise ValueError(msg)
        return values


class ProblemTaxonomy(_ProblemModel):
    """Versioned domain vocabulary used by models, SQL marts, and reports."""

    schema_version: int = Field(default=1, ge=1)
    taxonomy_id: str = Field(pattern=_CATEGORY_ID_PATTERN)
    version: str = Field(pattern=_VERSION_PATTERN)
    domain_ru: str = Field(min_length=1)
    locale: str = "ru"
    categories: tuple[ProblemCategory, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_unique_categories(self) -> ProblemTaxonomy:
        identifiers = [category.category_id for category in self.categories]
        names = [category.name_ru.casefold() for category in self.categories]
        if len(identifiers) != len(set(identifiers)):
            msg = "taxonomy category_id values must be unique"
            raise ValueError(msg)
        if len(names) != len(set(names)):
            msg = "taxonomy category names must be unique"
            raise ValueError(msg)
        return self

    @property
    def category_ids(self) -> frozenset[str]:
        """Return stable identifiers for validation without exposing mutable state."""
        return frozenset(category.category_id for category in self.categories)


class EvidenceSpan(_ProblemModel):
    """Character offsets into restricted feedback text; the text itself is not copied."""

    start: int = Field(ge=0)
    end: int = Field(gt=0)

    @model_validator(mode="after")
    def require_forward_span(self) -> EvidenceSpan:
        if self.end <= self.start:
            msg = "evidence end must be greater than start"
            raise ValueError(msg)
        return self


class FeedbackProblemAssessment(_ProblemModel):
    """One category-level judgment for a feedback record with explicit uncertainty."""

    feedback_id: str = Field(min_length=1)
    taxonomy_id: str = Field(pattern=_CATEGORY_ID_PATTERN)
    taxonomy_version: str = Field(pattern=_VERSION_PATTERN)
    category_id: str = Field(pattern=_CATEGORY_ID_PATTERN)
    stance: FeedbackStance
    category_probability: float = Field(ge=0, le=1)
    stance_probability: float = Field(ge=0, le=1)
    method: AssessmentMethod
    evidence_spans: tuple[EvidenceSpan, ...] = ()
    model_name: str | None = None
    model_revision: str | None = None
    review_status: ReviewStatus = ReviewStatus.UNREVIEWED
    reviewer_id: str | None = None
    assessed_at: datetime

    @model_validator(mode="after")
    def validate_review_and_evidence(self) -> FeedbackProblemAssessment:
        model_methods = {
            AssessmentMethod.EMBEDDING,
            AssessmentMethod.CLUSTER,
            AssessmentMethod.LLM,
            AssessmentMethod.HYBRID,
        }
        if self.method in model_methods and not (self.model_name or "").strip():
            msg = "model-based assessments require model_name"
            raise ValueError(msg)
        if self.method in model_methods and not (self.model_revision or "").strip():
            msg = "model-based assessments require model_revision"
            raise ValueError(msg)
        if self.method == AssessmentMethod.MANUAL and not (self.reviewer_id or "").strip():
            msg = "manual assessments require reviewer_id"
            raise ValueError(msg)
        if self.review_status != ReviewStatus.UNREVIEWED and not (self.reviewer_id or "").strip():
            msg = "reviewed assessments require reviewer_id"
            raise ValueError(msg)
        ordered = sorted(self.evidence_spans, key=lambda span: (span.start, span.end))
        if list(self.evidence_spans) != ordered:
            msg = "evidence spans must be sorted"
            raise ValueError(msg)
        if any(left.end > right.start for left, right in pairwise(ordered)):
            msg = "evidence spans cannot overlap"
            raise ValueError(msg)
        return self


class ProblemAssessmentBatch(_ProblemModel):
    """Checksum-bound multi-label assessments for one canonical feedback artifact."""

    schema_version: int = Field(default=1, ge=1)
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    taxonomy_id: str = Field(pattern=_CATEGORY_ID_PATTERN)
    taxonomy_version: str = Field(pattern=_VERSION_PATTERN)
    generated_at: datetime
    assessments: tuple[FeedbackProblemAssessment, ...]

    @model_validator(mode="after")
    def validate_assessment_bindings(self) -> ProblemAssessmentBatch:
        keys: set[tuple[str, str]] = set()
        for assessment in self.assessments:
            if (assessment.taxonomy_id, assessment.taxonomy_version) != (self.taxonomy_id, self.taxonomy_version):
                msg = "assessment taxonomy does not match batch"
                raise ValueError(msg)
            key = (assessment.feedback_id, assessment.category_id)
            if key in keys:
                msg = "batch contains duplicate feedback/category assessment"
                raise ValueError(msg)
            keys.add(key)
        return self


class CategoryDetectionRule(_ProblemModel):
    """Literal Russian stems used to produce conservative category candidates."""

    category_id: str = Field(pattern=_CATEGORY_ID_PATTERN)
    evidence_terms: tuple[str, ...] = Field(min_length=1)

    @field_validator("evidence_terms")
    @classmethod
    def normalize_terms(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(dict.fromkeys(value.strip().casefold() for value in values if value.strip()))
        if not normalized:
            msg = "category detection terms cannot be empty"
            raise ValueError(msg)
        return normalized


class ProblemDetectionRules(_ProblemModel):
    """Versioned, deterministic baseline for hybrid problem candidate generation."""

    schema_version: int = Field(default=1, ge=1)
    ruleset_id: str = Field(pattern=_CATEGORY_ID_PATTERN)
    version: str = Field(pattern=_VERSION_PATTERN)
    taxonomy_id: str = Field(pattern=_CATEGORY_ID_PATTERN)
    taxonomy_version: str = Field(pattern=_VERSION_PATTERN)
    complaint_rating_max: int = Field(default=3, ge=1, le=5)
    praise_rating_min: int = Field(default=4, ge=1, le=5)
    complaint_terms: tuple[str, ...] = Field(min_length=1)
    praise_terms: tuple[str, ...] = Field(min_length=1)
    categories: tuple[CategoryDetectionRule, ...] = Field(min_length=1)

    @field_validator("complaint_terms", "praise_terms")
    @classmethod
    def normalize_stance_terms(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(dict.fromkeys(value.strip().casefold() for value in values if value.strip()))
        if not normalized:
            msg = "stance detection terms cannot be empty"
            raise ValueError(msg)
        return normalized

    @model_validator(mode="after")
    def validate_rules(self) -> ProblemDetectionRules:
        if self.complaint_rating_max >= self.praise_rating_min:
            msg = "complaint and praise rating ranges must not overlap"
            raise ValueError(msg)
        identifiers = [category.category_id for category in self.categories]
        if len(identifiers) != len(set(identifiers)):
            msg = "problem detection category rules must be unique"
            raise ValueError(msg)
        return self


def load_problem_taxonomy(path: Path) -> ProblemTaxonomy:
    """Load a strict UTF-8 JSON taxonomy without source feedback data."""
    return ProblemTaxonomy.model_validate(json.loads(path.read_text(encoding="utf-8")))


def load_problem_detection_rules(path: Path) -> ProblemDetectionRules:
    """Load strict deterministic rules without executing patterns from configuration."""
    return ProblemDetectionRules.model_validate(json.loads(path.read_text(encoding="utf-8")))


def validate_problem_detection_rules(
    rules: ProblemDetectionRules,
    taxonomy: ProblemTaxonomy,
) -> ProblemDetectionRules:
    """Require rules to cover active categories of one exact taxonomy version."""
    if (rules.taxonomy_id, rules.taxonomy_version) != (taxonomy.taxonomy_id, taxonomy.version):
        msg = "problem detection rules do not match taxonomy"
        raise ValueError(msg)
    configured = {rule.category_id for rule in rules.categories}
    active = {category.category_id for category in taxonomy.categories if category.active}
    if configured != active:
        missing = ", ".join(sorted(active - configured)) or "none"
        unknown = ", ".join(sorted(configured - active)) or "none"
        msg = f"problem detection rule coverage mismatch; missing={missing}; unknown={unknown}"
        raise ValueError(msg)
    return rules


def validate_assessment_batch(
    batch: ProblemAssessmentBatch,
    taxonomy: ProblemTaxonomy,
) -> ProblemAssessmentBatch:
    """Bind a generated batch to an exact taxonomy and reject unknown categories."""
    if (batch.taxonomy_id, batch.taxonomy_version) != (taxonomy.taxonomy_id, taxonomy.version):
        msg = "assessment batch does not match taxonomy"
        raise ValueError(msg)
    unknown = {assessment.category_id for assessment in batch.assessments} - taxonomy.category_ids
    if unknown:
        msg = f"assessment batch contains unknown categories: {', '.join(sorted(unknown))}"
        raise ValueError(msg)
    return batch
