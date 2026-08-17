from pathlib import Path

import numpy as np
import pytest

from src.ml.outlier_reassignment import (
    OutlierDecision,
    OutlierDecisionReason,
    OutlierReassignmentConfig,
    OutlierReassignmentManifest,
    _normalized_chunk,
    reassign_outliers,
)
from src.ml.topic_representation import TopicRepresentationManifest
from tests.test_topic_representation import FakeTopicBackend, _build_inputs, _run


def _build_pipeline(
    tmp_path: Path,
    *,
    labels_values: list[int] | None = None,
) -> tuple[tuple[Path, ...], Path]:
    paths = _build_inputs(tmp_path, labels_values=labels_values)
    topics_dir = tmp_path / "topics"
    _run(paths, topics_dir, FakeTopicBackend())
    return paths, topics_dir / "topic-representation-manifest.json"


def _config(**updates: object) -> OutlierReassignmentConfig:
    defaults = {
        "similarity_threshold": 0.8,
        "single_topic_similarity_threshold": 0.8,
        "margin_threshold": 0.05,
        "minimum_topic_size": 1,
        "minimum_topic_mean_probability": 0.1,
        "centroid_member_minimum_probability": 0.5,
        "minimum_centroid_members": 1,
        "minimum_centroid_cohesion": 0.1,
        "maximum_centroid_similarity": 0.99,
        "maximum_topic_expansion_ratio": 1.0,
        "maximum_global_reassignment_share": 1.0,
        "block_size": 2,
    }
    defaults.update(updates)
    return OutlierReassignmentConfig.model_validate(defaults)


def _run_reassignment(
    paths: tuple[Path, ...],
    topic_manifest: Path,
    output: Path,
    *,
    config: OutlierReassignmentConfig | None = None,
    force: bool = False,
) -> OutlierReassignmentManifest:
    corpus, embeddings, labels, probabilities, corpus_manifest, clustering_manifest = paths
    return reassign_outliers(
        embeddings,
        labels,
        probabilities,
        corpus,
        corpus_manifest,
        clustering_manifest,
        topic_manifest,
        output,
        config=config or _config(),
        force=force,
    )


def test_reassign_outliers_preserves_sources_and_writes_auditable_outputs(tmp_path: Path) -> None:
    paths, topic_manifest = _build_pipeline(tmp_path)
    source_labels = paths[2].read_bytes()
    output = tmp_path / "reassignment"

    manifest = _run_reassignment(paths, topic_manifest, output)

    final_labels = np.load(output / "final-cluster-labels.npy", allow_pickle=False)
    confidence = np.load(output / "final-cluster-confidence.npy", allow_pickle=False)
    decisions = [
        OutlierDecision.model_validate_json(line)
        for line in (output / "outlier-decisions.jsonl").read_text().splitlines()
    ]
    assert final_labels.tolist() == [0, 0, 0, 1, 1, 1, 1]
    assert final_labels.dtype == np.int64
    assert confidence.dtype == np.float32
    assert all(decision.reassigned for decision in decisions)
    assert [decision.record_index for decision in decisions] == [5, 6]
    assert all(decision.best_topic == 1 for decision in decisions)
    assert paths[2].read_bytes() == source_labels
    assert manifest.metrics.original_outliers == 2
    assert manifest.metrics.reassigned_outliers == 2
    assert manifest.original_confidence_source == "hdbscan_probability"
    assert manifest.reassigned_confidence_source == "embedding_cosine_similarity"
    assert OutlierReassignmentManifest.model_validate_json(
        (output / "outlier-reassignment-manifest.json").read_text(),
    ) == manifest
    report = (output / "outlier-reassignment-report.md").read_text()
    assert "Private outlier" not in report
    assert "author-" not in report


def test_reassignment_keeps_low_similarity_or_disabled_outliers(tmp_path: Path) -> None:
    paths, topic_manifest = _build_pipeline(tmp_path)
    strict_output = tmp_path / "strict"
    strict = _run_reassignment(
        paths,
        topic_manifest,
        strict_output,
        config=_config(similarity_threshold=1.0, single_topic_similarity_threshold=1.0),
    )
    assert strict.metrics.reassigned_outliers == 0
    decisions = [
        OutlierDecision.model_validate_json(line)
        for line in (strict_output / "outlier-decisions.jsonl").read_text().splitlines()
    ]
    assert {decision.reason for decision in decisions} == {OutlierDecisionReason.BELOW_SIMILARITY}

    disabled_output = tmp_path / "disabled"
    disabled = _run_reassignment(paths, topic_manifest, disabled_output, config=_config(enabled=False))
    assert disabled.metrics.reassigned_outliers == 0
    disabled_decisions = [
        OutlierDecision.model_validate_json(line)
        for line in (disabled_output / "outlier-decisions.jsonl").read_text().splitlines()
    ]
    assert {decision.reason for decision in disabled_decisions} == {OutlierDecisionReason.DISABLED}

    margin_output = tmp_path / "margin"
    margin = _run_reassignment(paths, topic_manifest, margin_output, config=_config(margin_threshold=2.0))
    assert margin.metrics.reassigned_outliers == 0
    margin_decisions = [
        OutlierDecision.model_validate_json(line)
        for line in (margin_output / "outlier-decisions.jsonl").read_text().splitlines()
    ]
    assert {decision.reason for decision in margin_decisions} == {OutlierDecisionReason.INSUFFICIENT_MARGIN}


def test_reassignment_applies_global_and_topic_growth_limits(tmp_path: Path) -> None:
    paths, topic_manifest = _build_pipeline(tmp_path)
    output = tmp_path / "limited"
    manifest = _run_reassignment(
        paths,
        topic_manifest,
        output,
        config=_config(maximum_global_reassignment_share=0.5),
    )
    decisions = [
        OutlierDecision.model_validate_json(line)
        for line in (output / "outlier-decisions.jsonl").read_text().splitlines()
    ]
    assert manifest.metrics.reassigned_outliers == 1
    assert sum(decision.reason == OutlierDecisionReason.GLOBAL_LIMIT for decision in decisions) == 1

    topic_output = tmp_path / "topic-limited"
    topic_limited = _run_reassignment(
        paths,
        topic_manifest,
        topic_output,
        config=_config(maximum_topic_expansion_ratio=0.5),
    )
    assert topic_limited.metrics.reassigned_outliers == 1


def test_reassignment_handles_no_outliers_only_outliers_and_one_topic(tmp_path: Path) -> None:
    no_outlier_paths, no_outlier_topics = _build_pipeline(
        tmp_path / "no-outliers",
        labels_values=[0, 0, 0, 1, 1, 1, 1],
    )
    no_outliers = _run_reassignment(no_outlier_paths, no_outlier_topics, tmp_path / "no-outliers-result")
    assert no_outliers.metrics.original_outliers == 0
    assert (tmp_path / "no-outliers-result" / "outlier-decisions.jsonl").read_text() == ""

    only_outlier_paths, only_outlier_topics = _build_pipeline(
        tmp_path / "only-outliers",
        labels_values=[-1] * 7,
    )
    only_outliers = _run_reassignment(only_outlier_paths, only_outlier_topics, tmp_path / "only-outliers-result")
    assert only_outliers.metrics.eligible_topics == 0
    assert only_outliers.metrics.remaining_outliers == 7

    one_topic_paths, one_topic_manifest = _build_pipeline(
        tmp_path / "one-topic",
        labels_values=[0, 0, 0, 0, 0, -1, -1],
    )
    one_topic = _run_reassignment(
        one_topic_paths,
        one_topic_manifest,
        tmp_path / "one-topic-result",
        config=_config(similarity_threshold=0.0, single_topic_similarity_threshold=0.0),
    )
    assert one_topic.metrics.eligible_topics == 1
    assert one_topic.metrics.reassigned_outliers == 2


def test_reassignment_rejects_limited_topics_and_protects_outputs(tmp_path: Path) -> None:
    paths, topic_manifest_path = _build_pipeline(tmp_path)
    topic_manifest = TopicRepresentationManifest.model_validate_json(topic_manifest_path.read_text())
    limited = topic_manifest.model_copy(update={"topics": 1, "omitted_topics": 1})
    topic_manifest_path.write_text(limited.model_dump_json(), encoding="utf-8")
    with pytest.raises(ValueError, match="complete topic representation"):
        _run_reassignment(paths, topic_manifest_path, tmp_path / "invalid")

    paths, topic_manifest_path = _build_pipeline(tmp_path / "protected")
    output = tmp_path / "protected-result"
    _run_reassignment(paths, topic_manifest_path, output)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _run_reassignment(paths, topic_manifest_path, output)
    assert _run_reassignment(paths, topic_manifest_path, output, force=True).metrics.reassigned_outliers == 2


def test_reassignment_rejects_nonfinite_and_zero_embedding_vectors() -> None:
    with pytest.raises(ValueError, match="NaN or infinite"):
        _normalized_chunk(np.asarray([[float("nan"), 1.0]], dtype=np.float32))
    with pytest.raises(ValueError, match="zero vectors"):
        _normalized_chunk(np.asarray([[0.0, 0.0]], dtype=np.float32))
