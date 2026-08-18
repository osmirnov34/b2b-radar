from pathlib import Path

import numpy as np
import pytest

from src.ml.clustering import HDBSCANConfig
from src.ml.dimensionality_reduction import UMAPConfig
from src.ml.evaluation import (
    BootstrapRun,
    ClusterMatch,
    EvaluationConfig,
    EvaluationManifest,
    EvaluationStatus,
    GeometryMetrics,
    ManualReviewRecord,
    deterministic_evaluation_indices,
    evaluate_topics,
    match_clusters,
)
from tests.test_outlier_reassignment import _build_pipeline, _run_reassignment


class FakeGeometryBackend:
    def evaluate(self, vectors: np.ndarray, labels: np.ndarray) -> GeometryMetrics:
        del vectors
        clusters = len({int(value) for value in labels if value >= 0})
        return GeometryMetrics(
            evaluated_records=int(np.count_nonzero(labels >= 0)),
            clusters=clusters,
            silhouette=0.7 if clusters >= 2 else None,
            davies_bouldin=0.5 if clusters >= 2 else None,
            calinski_harabasz=10.0 if clusters >= 2 else None,
            mean_intra_cluster_similarity=0.9 if clusters else None,
            maximum_centroid_similarity=0.2 if clusters >= 2 else None,
        )


class FakeStabilityBackend:
    def evaluate(
        self,
        vectors: np.ndarray,
        labels: np.ndarray,
        config: EvaluationConfig,
        umap_config: UMAPConfig,
        hdbscan_config: HDBSCANConfig,
    ) -> tuple[list[BootstrapRun], list[ClusterMatch]]:
        del vectors, labels, config, umap_config, hdbscan_config
        return [BootstrapRun(run=0, records=7, ari=0.9, nmi=0.9)], []


def _run_evaluation(tmp_path: Path, *, annotations: Path | None = None) -> tuple[EvaluationManifest, Path]:
    paths, topic_manifest = _build_pipeline(tmp_path)
    reassignment_dir = tmp_path / "reassignment"
    _run_reassignment(paths, topic_manifest, reassignment_dir)
    corpus, embeddings, labels, _, corpus_manifest, clustering_manifest = paths
    output = tmp_path / "evaluation"
    manifest = evaluate_topics(
        embeddings,
        corpus,
        labels,
        reassignment_dir / "final-cluster-labels.npy",
        reassignment_dir / "final-cluster-confidence.npy",
        corpus_manifest,
        clustering_manifest,
        topic_manifest,
        reassignment_dir / "outlier-reassignment-manifest.json",
        output,
        config=EvaluationConfig(
            geometry_sample_size=7,
            bootstrap_runs=0,
            manual_topics=2,
            manual_examples_per_topic=2,
            manual_outliers=1,
            minimum_manual_annotations=2,
            validation_completed=annotations is not None,
        ),
        manual_annotations_path=annotations,
        geometry_factory=FakeGeometryBackend,
        stability_factory=FakeStabilityBackend,
    )
    return manifest, output


def test_evaluation_writes_preliminary_safe_report_and_sensitive_local_sample(tmp_path: Path) -> None:
    manifest, output = _run_evaluation(tmp_path)

    assert manifest.warnings == ["manual review is missing", "validation evaluation is not completed"]
    metrics = (output / "evaluation-metrics.json").read_text()
    report = (output / "evaluation-report.md").read_text()
    samples = [
        ManualReviewRecord.model_validate_json(line)
        for line in (output / "manual-review-sample.jsonl").read_text().splitlines()
    ]
    assert '"status":"pass_with_warnings"' in metrics.replace(" ", "").replace("\n", "")
    assert "Private" not in report
    assert samples
    assert any(sample.sample_kind == "reassigned" for sample in samples)
    assert EvaluationManifest.model_validate_json((output / "evaluation-manifest.json").read_text()) == manifest


def test_manual_review_and_validation_allow_final_pass(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations.jsonl"
    annotations.write_text(
        (
            '{"record_index":0,"topic_matches":true,"topic_clear":true,"business_relevant":true}\n'
            '{"record_index":5,"topic_matches":true,"topic_clear":true,"business_relevant":true,'
            '"reassignment_correct":true}\n'
        ),
    )
    _, output = _run_evaluation(tmp_path, annotations=annotations)
    content = (output / "evaluation-metrics.json").read_text()
    assert f'"status": "{EvaluationStatus.PASS}"' in content
    assert '"preliminary": false' in content


def test_cluster_matching_detects_splits_and_rejects_shape_mismatch() -> None:
    pytest.importorskip("scipy")
    reference = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    candidate = np.asarray([2, 2, 3, 2, 4, 4], dtype=np.int64)
    matches = match_clusters(reference, candidate, run=0)
    assert any(item.split_detected for item in matches)
    assert any(item.merge_detected for item in matches)
    with pytest.raises(ValueError, match="identical shapes"):
        match_clusters(reference, candidate[:-1], run=0)


def test_evaluation_sampling_is_reproducible() -> None:
    first = deterministic_evaluation_indices(100, 10, 42)
    assert first == deterministic_evaluation_indices(100, 10, 42)
    assert first != deterministic_evaluation_indices(100, 10, 43)
