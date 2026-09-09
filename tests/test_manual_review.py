import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.ml.evaluation import EvaluationConfig, EvaluationManifest, ManualAnnotation, ManualReviewRecord
from src.ml.manual_review import (
    load_manual_review_bundle,
    reviewer_agreement,
    save_manual_annotations,
    summarize_manual_annotations,
    validate_manual_annotations,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _review_run(tmp_path: Path) -> tuple[Path, list[ManualAnnotation]]:
    run = tmp_path / "run"
    evaluation_dir = run / "12-evaluation"
    evaluation_dir.mkdir(parents=True)
    sample = [
        ManualReviewRecord(record_index=1, topic_id=0, sample_kind="topic_member", text="private one", confidence=0.8),
        ManualReviewRecord(record_index=3, topic_id=0, sample_kind="reassigned", text="private two", confidence=0.9),
    ]
    sample_path = evaluation_dir / "manual-review-sample.jsonl"
    sample_path.write_text("".join(f"{item.model_dump_json()}\n" for item in sample), encoding="utf-8")
    template_path = evaluation_dir / "manual-review-template.json"
    template_path.write_text("{}\n", encoding="utf-8")
    placeholder_sha256 = "a" * 64
    evaluation = EvaluationManifest(
        corpus_manifest_path=str(run / "07-corpus/corpus-manifest.json"),
        corpus_manifest_sha256=placeholder_sha256,
        clustering_manifest_path=str(run / "09-clustering/clustering-manifest.json"),
        clustering_manifest_sha256=placeholder_sha256,
        topic_manifest_path=str(run / "10-topics/topic-representation-manifest.json"),
        topic_manifest_sha256=placeholder_sha256,
        reassignment_manifest_path=str(run / "11-reassignment/outlier-reassignment-manifest.json"),
        reassignment_manifest_sha256=placeholder_sha256,
        config=EvaluationConfig(),
        metrics_path=str(evaluation_dir / "evaluation-metrics.json"),
        metrics_sha256=placeholder_sha256,
        topic_evaluation_path=str(evaluation_dir / "topic-evaluation.jsonl"),
        topic_evaluation_sha256=placeholder_sha256,
        cluster_matching_path=str(evaluation_dir / "cluster-matching.jsonl"),
        cluster_matching_sha256=placeholder_sha256,
        manual_review_sample_path=str(sample_path),
        manual_review_sample_sha256=_sha256(sample_path),
        manual_review_template_path=str(template_path),
        manual_review_template_sha256=_sha256(template_path),
        warnings=["manual review pending"],
        created_at=datetime(2025, 1, 1, tzinfo=UTC),
    )
    (evaluation_dir / "evaluation-manifest.json").write_text(
        f"{evaluation.model_dump_json()}\n",
        encoding="utf-8",
    )
    (run / "run-manifest.json").write_text(
        json.dumps({"schema_version": 1, "run_id": "manual-run", "status": "awaiting_review"}),
        encoding="utf-8",
    )
    annotations = [
        ManualAnnotation(
            record_index=1,
            topic_matches=True,
            topic_clear=True,
            business_relevant=False,
            reviewer="reviewer-a",
        ),
        ManualAnnotation(
            record_index=3,
            topic_matches=False,
            topic_clear=True,
            business_relevant=True,
            reassignment_correct=False,
            contains_sensitive_data=True,
            reviewer="reviewer-a",
        ),
    ]
    return run, annotations


def test_manual_review_load_resume_summarize_and_save_outside_run(tmp_path: Path) -> None:
    run, annotations = _review_run(tmp_path)
    existing = tmp_path / "existing.jsonl"
    existing.write_text(f"{annotations[0].model_dump_json()}\n", encoding="utf-8")

    bundle = load_manual_review_bundle(run, existing)
    summary = summarize_manual_annotations(bundle)
    complete_bundle = bundle.model_copy(update={"annotations": tuple(annotations)})
    manifest = save_manual_annotations(complete_bundle, annotations, tmp_path / "reviews")

    assert len(bundle.sample) == 2
    assert len(bundle.annotations) == 1
    assert summary.coverage == 0.5
    assert summary.reviewers == ("reviewer-a",)
    assert manifest.annotations == 2
    assert manifest.classification == "restricted_manual_review"
    assert "private one" not in (tmp_path / "reviews/manual-annotations-manifest.json").read_text()
    with pytest.raises(FileExistsError, match="already exist"):
        save_manual_annotations(complete_bundle, annotations, tmp_path / "reviews")
    with pytest.raises(ValueError, match="outside the immutable run"):
        save_manual_annotations(complete_bundle, annotations, run / "annotations")


def test_manual_annotation_validation_enforces_sample_kind_identity_and_uniqueness(tmp_path: Path) -> None:
    run, annotations = _review_run(tmp_path)
    bundle = load_manual_review_bundle(run)

    with pytest.raises(ValueError, match="duplicate"):
        validate_manual_annotations(bundle.sample, [annotations[0], annotations[0]])
    with pytest.raises(ValueError, match="outside"):
        validate_manual_annotations(
            bundle.sample,
            [annotations[0].model_copy(update={"record_index": 999})],
        )
    with pytest.raises(ValueError, match="reviewer identity"):
        validate_manual_annotations(
            bundle.sample,
            [annotations[0].model_copy(update={"reviewer": ""})],
        )
    with pytest.raises(ValueError, match="sample kind"):
        validate_manual_annotations(
            bundle.sample,
            [annotations[1].model_copy(update={"reassignment_correct": None})],
        )


def test_reviewer_agreement_uses_only_overlapping_binary_judgments(tmp_path: Path) -> None:
    _, annotations = _review_run(tmp_path)
    second = tuple(
        annotation.model_copy(
            update={
                "reviewer": "reviewer-b",
                "business_relevant": not annotation.business_relevant,
            },
        )
        for annotation in annotations
    )

    agreement = reviewer_agreement({"reviewer-a": tuple(annotations), "reviewer-b": second})

    assert agreement.reviewer_pairs == 1
    assert agreement.overlapping_records == 2
    assert agreement.compared_judgments == 7
    assert agreement.matching_judgments == 5
    assert agreement.observed_agreement == pytest.approx(5 / 7)
    assert "not chance-corrected" in agreement.interpretation


def test_manual_review_rejects_tampered_sample(tmp_path: Path) -> None:
    run, _ = _review_run(tmp_path)
    (run / "12-evaluation/manual-review-sample.jsonl").write_text("tampered\n", encoding="utf-8")

    with pytest.raises(ValueError, match="checksum mismatch"):
        load_manual_review_bundle(run)


def test_manual_review_resume_verifies_saved_annotation_binding(tmp_path: Path) -> None:
    run, annotations = _review_run(tmp_path)
    bundle = load_manual_review_bundle(run)
    review_dir = tmp_path / "reviews"
    save_manual_annotations(bundle, annotations, review_dir)

    resumed = load_manual_review_bundle(run, review_dir / "manual-annotations.jsonl")
    assert resumed.annotations == tuple(annotations)

    annotations_path = review_dir / "manual-annotations.jsonl"
    annotations_path.write_text(f"{annotations[0].model_dump_json()}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="do not match"):
        load_manual_review_bundle(run, annotations_path)
