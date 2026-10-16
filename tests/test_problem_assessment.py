import hashlib
import json
from pathlib import Path

import pytest

from src.analytics import (
    AssessmentMethod,
    FeedbackProblemAssessment,
    FeedbackStance,
    ProblemAssessmentArtifactManifest,
    load_problem_detection_rules,
    load_problem_taxonomy,
    validate_problem_detection_rules,
)
from src.operations.feedback_preparation import prepare_yandex_feedback
from src.operations.problem_assessment import assess_feedback_problems

PROJECT_ROOT = Path(__file__).parents[1]
TAXONOMY = PROJECT_ROOT / "configs/dental-problem-taxonomy.v1.json"
RULES = PROJECT_ROOT / "configs/dental-problem-rules.v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prepared_feedback(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "reviews.jsonl"
    rows = [
        {
            "review_id": "aaaaaaaaaaaaaaaa",
            "review_text": "Ждала в очереди очень долго, администратор грубил.",
            "rating": 1,
            "published_at": "2025-04-03",
            "org_id": "123456789",
            "org_name": "Клиника",
            "org_url": "https://yandex.ru/maps/org/123456789/reviews/",
            "search_query": "стоматология",
        },
        {
            "review_id": "bbbbbbbbbbbbbbbb",
            "review_text": "Лечение прошло отлично и без боли.",
            "rating": 5,
            "published_at": "2025-04-04",
            "org_id": "123456789",
            "org_name": "Клиника",
            "org_url": "https://yandex.ru/maps/org/123456789/reviews/",
            "search_query": "стоматология",
        },
    ]
    source.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    feedback = tmp_path / "feedback.jsonl"
    manifest = tmp_path / "feedback.manifest.json"
    prepare_yandex_feedback(source, feedback, manifest)
    return feedback, manifest


def _topic_artifact(tmp_path: Path) -> tuple[Path, Path]:
    assignment = tmp_path / "topics.jsonl"
    assignment.write_text(
        json.dumps(
            {
                "feedback_id": "yandex_maps:aaaaaaaaaaaaaaaa",
                "source": "yandex_maps",
                "source_record_id": "aaaaaaaaaaaaaaaa",
                "rating": 1,
                "published_on": "2025-04-03",
                "organization_id": "123456789",
                "organization_name": "Клиника",
                "organization_url": "https://yandex.ru/maps/org/123456789/reviews/",
                "search_query": "стоматология",
                "corpus_id": f"corpus:{'a' * 64}",
                "topic_id": 0,
                "topic_name": "ожидание / очередь",
                "confidence": 0.9,
                "reassigned": False,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "topics.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "feedback_path": "feedback.jsonl",
                "feedback_sha256": "1" * 64,
                "feedback_manifest_path": "feedback.manifest.json",
                "feedback_manifest_sha256": "2" * 64,
                "corpus_path": "corpus.jsonl",
                "corpus_sha256": "3" * 64,
                "reassignment_manifest_path": "reassignment.json",
                "reassignment_manifest_sha256": "4" * 64,
                "topic_manifest_path": "topic.json",
                "topic_manifest_sha256": "5" * 64,
                "output_path": str(assignment),
                "output_sha256": _sha256(assignment),
                "adapted_at": "2026-10-16T00:00:00Z",
                "feedback_records": 2,
                "corpus_records": 1,
                "matched_records": 1,
                "unmatched_feedback_records": 1,
                "assigned_records": 1,
                "outlier_records": 0,
                "reassigned_records": 0,
                "topics": 1,
            },
        )
        + "\n",
        encoding="utf-8",
    )
    return assignment, manifest


def test_detection_rules_cover_exact_taxonomy() -> None:
    taxonomy = load_problem_taxonomy(TAXONOMY)
    rules = validate_problem_detection_rules(load_problem_detection_rules(RULES), taxonomy)

    assert {rule.category_id for rule in rules.categories} == taxonomy.category_ids


@pytest.mark.parametrize(
    ("text", "rating"),
    [
        ("Лечили аккуратно, было не больно.", 5),
        ("Цена оказалась не дорогой, всё объяснили.", 5),
        ("Приняли вовремя, не пришлось ждать.", 5),
        ("Лишние процедуры не навязывали.", 5),
    ],
)
def test_positive_negation_is_not_classified_as_mixed(text: str, rating: int, tmp_path: Path) -> None:
    source = tmp_path / "reviews.jsonl"
    source.write_text(
        json.dumps(
            {
                "review_id": "cccccccccccccccc",
                "review_text": text,
                "rating": rating,
                "published_at": "2025-04-05",
                "org_id": "123456789",
                "org_name": "Клиника",
                "org_url": "https://yandex.ru/maps/org/123456789/reviews/",
                "search_query": "стоматология",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    feedback = tmp_path / "feedback.jsonl"
    feedback_manifest = tmp_path / "feedback.manifest.json"
    prepare_yandex_feedback(source, feedback, feedback_manifest)
    output = tmp_path / "assessments.jsonl"

    assess_feedback_problems(feedback, feedback_manifest, TAXONOMY, RULES, output)

    assessments = [
        FeedbackProblemAssessment.model_validate_json(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert assessments
    assert {item.stance for item in assessments} == {FeedbackStance.PRAISE}


def test_negative_phrase_shadows_embedded_positive_term(tmp_path: Path) -> None:
    source = tmp_path / "reviews.jsonl"
    source.write_text(
        json.dumps(
            {
                "review_id": "dddddddddddddddd",
                "review_text": "Лечение прошло плохо, клинику не рекомендую.",
                "rating": 1,
                "published_at": "2025-04-05",
                "org_id": "123456789",
                "org_name": "Клиника",
                "org_url": "https://yandex.ru/maps/org/123456789/reviews/",
                "search_query": "стоматология",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    feedback = tmp_path / "feedback.jsonl"
    feedback_manifest = tmp_path / "feedback.manifest.json"
    prepare_yandex_feedback(source, feedback, feedback_manifest)
    output = tmp_path / "assessments.jsonl"

    assess_feedback_problems(feedback, feedback_manifest, TAXONOMY, RULES, output)

    assessments = [
        FeedbackProblemAssessment.model_validate_json(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert assessments
    assert {item.stance for item in assessments} == {FeedbackStance.COMPLAINT}


def test_problem_assessment_emits_evidence_without_private_text(tmp_path: Path) -> None:
    feedback, feedback_manifest = _prepared_feedback(tmp_path)
    topics, topics_manifest = _topic_artifact(tmp_path)
    output = tmp_path / "assessments.jsonl"

    result = assess_feedback_problems(
        feedback,
        feedback_manifest,
        TAXONOMY,
        RULES,
        output,
        topic_assignments_path=topics,
        topic_manifest_path=topics_manifest,
    )

    assessments = [
        FeedbackProblemAssessment.model_validate_json(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]
    waiting = next(item for item in assessments if item.category_id == "waiting_time")
    pain = next(
        item
        for item in assessments
        if item.feedback_id == "yandex_maps:bbbbbbbbbbbbbbbb" and item.category_id == "pain_and_comfort"
    )
    assert waiting.method == AssessmentMethod.HYBRID
    assert waiting.stance == FeedbackStance.COMPLAINT
    assert pain.method == AssessmentMethod.RULE
    assert pain.stance == FeedbackStance.PRAISE
    assert all(item.review_status == "unreviewed" for item in assessments)
    assert "Ждала" not in output.read_text(encoding="utf-8")
    assert result.assessed_feedback_records == 2
    assert result.hybrid_assessments >= 1
    persisted = ProblemAssessmentArtifactManifest.model_validate_json(
        output.with_suffix(".jsonl.manifest.json").read_text(encoding="utf-8"),
    )
    assert persisted == result


def test_problem_assessment_rejects_topic_checksum_mismatch(tmp_path: Path) -> None:
    feedback, feedback_manifest = _prepared_feedback(tmp_path)
    topics, topics_manifest = _topic_artifact(tmp_path)
    topics.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="do not match"):
        assess_feedback_problems(
            feedback,
            feedback_manifest,
            TAXONOMY,
            RULES,
            tmp_path / "assessments.jsonl",
            topic_assignments_path=topics,
            topic_manifest_path=topics_manifest,
        )
