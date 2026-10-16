from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.analytics import (
    AssessmentMethod,
    EvidenceSpan,
    FeedbackProblemAssessment,
    FeedbackStance,
    ProblemAssessmentBatch,
    ProblemTaxonomy,
    ReviewStatus,
    load_problem_taxonomy,
    validate_assessment_batch,
)

PROJECT_ROOT = Path(__file__).parents[1]
TAXONOMY_PATH = PROJECT_ROOT / "configs/dental-problem-taxonomy.v1.json"


def _assessment(**updates: object) -> FeedbackProblemAssessment:
    data: dict[str, object] = {
        "feedback_id": "yandex_maps:7995cc14c8b92196",
        "taxonomy_id": "moscow_dentistry_problems",
        "taxonomy_version": "1.0.0",
        "category_id": "waiting_time",
        "stance": FeedbackStance.COMPLAINT,
        "category_probability": 0.91,
        "stance_probability": 0.88,
        "method": AssessmentMethod.HYBRID,
        "model_name": "problem-classifier",
        "model_revision": "revision-001",
        "evidence_spans": [{"start": 4, "end": 18}],
        "assessed_at": datetime(2026, 10, 16, tzinfo=UTC),
    }
    data.update(updates)
    return FeedbackProblemAssessment.model_validate(data)


def test_dental_taxonomy_is_strict_versioned_and_complete() -> None:
    taxonomy = load_problem_taxonomy(TAXONOMY_PATH)

    assert taxonomy.taxonomy_id == "moscow_dentistry_problems"
    assert taxonomy.version == "1.0.0"
    assert len(taxonomy.categories) == 10
    assert "treatment_quality" in taxonomy.category_ids
    assert "pricing_and_billing" in taxonomy.category_ids
    assert all(category.inclusion_rules for category in taxonomy.categories)
    assert all(category.exclusion_rules for category in taxonomy.categories)
    assert all(category.impact_hypotheses for category in taxonomy.categories)


def test_taxonomy_rejects_duplicate_category_identifiers() -> None:
    taxonomy = load_problem_taxonomy(TAXONOMY_PATH)
    duplicate = taxonomy.categories[0].model_copy(update={"name_ru": "Другое имя"})

    with pytest.raises(ValidationError, match="category_id values must be unique"):
        ProblemTaxonomy.model_validate(
            {**taxonomy.model_dump(), "categories": [*taxonomy.categories, duplicate]},
        )


def test_assessment_keeps_evidence_as_offsets_without_private_text() -> None:
    assessment = _assessment()

    assert assessment.evidence_spans == (EvidenceSpan(start=4, end=18),)
    assert "text" not in type(assessment).model_fields
    assert "quote" not in EvidenceSpan.model_fields


@pytest.mark.parametrize(
    "spans",
    [
        [{"start": 10, "end": 20}, {"start": 1, "end": 4}],
        [{"start": 1, "end": 10}, {"start": 9, "end": 15}],
    ],
)
def test_assessment_rejects_unsorted_or_overlapping_evidence(spans: list[dict[str, int]]) -> None:
    with pytest.raises(ValidationError, match="evidence spans"):
        _assessment(evidence_spans=spans)


def test_reviewed_assessment_requires_reviewer_identity() -> None:
    with pytest.raises(ValidationError, match="reviewer_id"):
        _assessment(review_status=ReviewStatus.ACCEPTED)

    reviewed = _assessment(review_status=ReviewStatus.ACCEPTED, reviewer_id="reviewer-01")
    assert reviewed.reviewer_id == "reviewer-01"


def test_model_assessment_requires_reproducible_model_identity() -> None:
    with pytest.raises(ValidationError, match="model_name"):
        _assessment(model_name=None)
    with pytest.raises(ValidationError, match="model_revision"):
        _assessment(model_revision=None)


def test_manual_assessment_requires_reviewer_identity() -> None:
    with pytest.raises(ValidationError, match="manual assessments require reviewer_id"):
        _assessment(method=AssessmentMethod.MANUAL, model_name=None, model_revision=None)


def test_batch_allows_multilabel_but_rejects_duplicate_category() -> None:
    first = _assessment()
    second = _assessment(category_id="staff_communication")
    batch = ProblemAssessmentBatch(
        artifact_sha256="a" * 64,
        taxonomy_id="moscow_dentistry_problems",
        taxonomy_version="1.0.0",
        generated_at=datetime(2026, 10, 16, tzinfo=UTC),
        assessments=(first, second),
    )

    assert len(batch.assessments) == 2
    with pytest.raises(ValidationError, match="duplicate feedback/category"):
        batch.model_copy(update={"assessments": (first, first)}).model_validate(
            {**batch.model_dump(), "assessments": [first, first]},
        )


def test_batch_rejects_assessment_from_another_taxonomy_version() -> None:
    assessment = _assessment(taxonomy_version="2.0.0")

    with pytest.raises(ValidationError, match="taxonomy does not match"):
        ProblemAssessmentBatch(
            artifact_sha256="a" * 64,
            taxonomy_id="moscow_dentistry_problems",
            taxonomy_version="1.0.0",
            generated_at=datetime(2026, 10, 16, tzinfo=UTC),
            assessments=(assessment,),
        )


def test_batch_must_use_categories_from_exact_taxonomy() -> None:
    taxonomy = load_problem_taxonomy(TAXONOMY_PATH)
    unknown = _assessment(category_id="unknown_category")
    batch = ProblemAssessmentBatch(
        artifact_sha256="a" * 64,
        taxonomy_id=taxonomy.taxonomy_id,
        taxonomy_version=taxonomy.version,
        generated_at=datetime(2026, 10, 16, tzinfo=UTC),
        assessments=(unknown,),
    )

    with pytest.raises(ValueError, match="unknown categories"):
        validate_assessment_batch(batch, taxonomy)
