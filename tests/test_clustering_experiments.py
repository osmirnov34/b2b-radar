import hashlib
from pathlib import Path

import pytest

from src.ml.clustering import HDBSCANConfig
from src.ml.clustering_experiments import (
    ClusteringGridConfig,
    ClusteringGridManifest,
    ClusterMatchingConfig,
    ClusterStabilityConfig,
    ClusterTransitionStatus,
    StabilityLevel,
    analyze_grid_stability,
    match_grid_clusters,
    run_clustering_grid,
)
from tests.test_clustering import FakeClusterer, _build_inputs


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class GridFactory:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def __call__(self, config: HDBSCANConfig) -> FakeClusterer:
        self.calls.append(config.min_cluster_size)
        if config.min_cluster_size == 2:
            return FakeClusterer([0, 0, 0, -1, 1, 1, -1, -1], [0.9, 0.8, 0.7, 0.0, 0.8, 0.7, 0.0, 0.0])
        return FakeClusterer([0, 0, 0, 0, -1, -1, -1, -1], [0.9, 0.8, 0.7, 0.6, 0.0, 0.0, 0.0, 0.0])


class SplitMergeFactory:
    def __call__(self, config: HDBSCANConfig) -> FakeClusterer:
        if config.min_cluster_size == 2:
            return FakeClusterer([0, 0, 0, 0, 1, 1, 1, 1], [0.8] * 8)
        return FakeClusterer([0, 0, 1, 1, 0, 0, 1, 1], [0.8] * 8)


class StableFactory:
    def __call__(self, _config: HDBSCANConfig) -> FakeClusterer:
        return FakeClusterer([0, 0, 0, 0, 1, 1, 1, 1], [0.8] * 8)


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "source-run" / "08-reduction"
    source.mkdir(parents=True)
    return _build_inputs(source)


def test_clustering_grid_runs_fixed_variants_and_publishes_manifest(tmp_path: Path) -> None:
    reduced, reduction_manifest, corpus_manifest = _inputs(tmp_path)
    output = tmp_path / "experiments" / "grid"
    factory = GridFactory()
    config = ClusteringGridConfig(
        min_cluster_sizes=(2, 3),
        base_config=HDBSCANConfig(min_cluster_size=250),
    )

    manifest = run_clustering_grid(
        reduced,
        reduction_manifest,
        corpus_manifest,
        output,
        config=config,
        clusterer_factory=factory,
    )

    assert factory.calls == [2, 3]
    assert [result.min_cluster_size for result in manifest.results] == [2, 3]
    assert [result.metrics.clusters for result in manifest.results] == [2, 1]
    assert [result.metrics.outlier_share for result in manifest.results] == [0.375, 0.5]
    assert manifest.reduced_sha256 == _sha256(reduced)
    persisted = ClusteringGridManifest.model_validate_json(
        (output / "clustering-grid-manifest.json").read_text(encoding="utf-8"),
    )
    assert persisted == manifest


def test_clustering_grid_resumes_verified_variants_without_refitting(tmp_path: Path) -> None:
    reduced, reduction_manifest, corpus_manifest = _inputs(tmp_path)
    output = tmp_path / "experiments" / "grid"
    config = ClusteringGridConfig(min_cluster_sizes=(2, 3))
    initial_factory = GridFactory()
    first = run_clustering_grid(
        reduced,
        reduction_manifest,
        corpus_manifest,
        output,
        config=config,
        clusterer_factory=initial_factory,
    )
    resume_factory = GridFactory()

    resumed = run_clustering_grid(
        reduced,
        reduction_manifest,
        corpus_manifest,
        output,
        config=config,
        clusterer_factory=resume_factory,
    )

    assert initial_factory.calls == [2, 3]
    assert resume_factory.calls == []
    assert resumed == first


def test_clustering_grid_blocks_partial_or_tampered_variants(tmp_path: Path) -> None:
    reduced, reduction_manifest, corpus_manifest = _inputs(tmp_path)
    output = tmp_path / "experiments" / "grid"
    partial = output / "min-cluster-size-2"
    partial.mkdir(parents=True)
    (partial / "partial.tmp").write_text("partial", encoding="utf-8")

    with pytest.raises(FileExistsError, match="manual inspection"):
        run_clustering_grid(
            reduced,
            reduction_manifest,
            corpus_manifest,
            output,
            config=ClusteringGridConfig(min_cluster_sizes=(2,)),
            clusterer_factory=GridFactory(),
        )

    (partial / "partial.tmp").unlink()
    run_clustering_grid(
        reduced,
        reduction_manifest,
        corpus_manifest,
        output,
        config=ClusteringGridConfig(min_cluster_sizes=(2,)),
        clusterer_factory=GridFactory(),
    )
    labels = partial / "cluster-labels.npy"
    with labels.open("ab") as target:
        target.write(b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        run_clustering_grid(
            reduced,
            reduction_manifest,
            corpus_manifest,
            output,
            config=ClusteringGridConfig(min_cluster_sizes=(2,)),
            clusterer_factory=GridFactory(),
        )


def test_clustering_grid_rejects_invalid_grid_and_source_run_output(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unique and strictly increasing"):
        ClusteringGridConfig(min_cluster_sizes=(100, 50, 100))
    reduced, reduction_manifest, corpus_manifest = _inputs(tmp_path)

    with pytest.raises(ValueError, match="outside the immutable source run"):
        run_clustering_grid(
            reduced,
            reduction_manifest,
            corpus_manifest,
            reduction_manifest.parent / "grid",
            config=ClusteringGridConfig(min_cluster_sizes=(2,)),
            clusterer_factory=GridFactory(),
        )


def test_grid_cluster_matching_detects_split_merge_and_primary_pairs(tmp_path: Path) -> None:
    reduced, reduction_manifest, corpus_manifest = _inputs(tmp_path)
    output = tmp_path / "experiments" / "grid"
    run_clustering_grid(
        reduced,
        reduction_manifest,
        corpus_manifest,
        output,
        config=ClusteringGridConfig(min_cluster_sizes=(2, 3)),
        clusterer_factory=SplitMergeFactory(),
    )

    manifest, transitions = match_grid_clusters(output / "clustering-grid-manifest.json")

    assert manifest.compared_pairs == [(2, 3)]
    assert manifest.records == 8
    assert len(transitions) == 4
    assert all(item.status == ClusterTransitionStatus.SPLIT_AND_MERGED for item in transitions)
    assert all(item.jaccard == 1 / 3 for item in transitions)
    assert all(item.source_retention == item.target_composition == 0.5 for item in transitions)
    assert sum(item.primary_match for item in transitions) == 1


def test_grid_cluster_matching_marks_disappeared_clusters_and_resumes(tmp_path: Path) -> None:
    reduced, reduction_manifest, corpus_manifest = _inputs(tmp_path)
    output = tmp_path / "experiments" / "grid"
    run_clustering_grid(
        reduced,
        reduction_manifest,
        corpus_manifest,
        output,
        config=ClusteringGridConfig(min_cluster_sizes=(2, 3)),
        clusterer_factory=GridFactory(),
    )

    first_manifest, first = match_grid_clusters(output / "clustering-grid-manifest.json")
    resumed_manifest, resumed = match_grid_clusters(output / "clustering-grid-manifest.json")

    assert resumed_manifest == first_manifest
    assert resumed == first
    disappeared = [item for item in first if item.status == ClusterTransitionStatus.DISAPPEARED]
    assert len(disappeared) == 1
    assert disappeared[0].source_cluster_id == 1
    assert disappeared[0].target_cluster_id is None


def test_grid_cluster_matching_threshold_and_checkpoint_integrity(tmp_path: Path) -> None:
    reduced, reduction_manifest, corpus_manifest = _inputs(tmp_path)
    output = tmp_path / "experiments" / "grid"
    run_clustering_grid(
        reduced,
        reduction_manifest,
        corpus_manifest,
        output,
        config=ClusteringGridConfig(min_cluster_sizes=(2, 3)),
        clusterer_factory=SplitMergeFactory(),
    )
    grid_path = output / "clustering-grid-manifest.json"
    _, transitions = match_grid_clusters(
        grid_path,
        config=ClusterMatchingConfig(minimum_overlap_share=0.6),
    )
    assert {item.status for item in transitions} == {
        ClusterTransitionStatus.DISAPPEARED,
        ClusterTransitionStatus.NEW,
    }
    (output / "cluster-transitions.jsonl").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="incompatible or damaged"):
        match_grid_clusters(grid_path, config=ClusterMatchingConfig(minimum_overlap_share=0.6))


def _matching_checkpoint(tmp_path: Path, factory: object, sizes: tuple[int, ...]) -> Path:
    reduced, reduction_manifest, corpus_manifest = _inputs(tmp_path)
    output = tmp_path / "experiments" / "grid"
    run_clustering_grid(
        reduced,
        reduction_manifest,
        corpus_manifest,
        output,
        config=ClusteringGridConfig(min_cluster_sizes=sizes),
        clusterer_factory=factory,  # type: ignore[arg-type]
    )
    match_grid_clusters(output / "clustering-grid-manifest.json")
    return output / "cluster-matching-manifest.json"


def test_grid_stability_builds_complete_stable_trajectories(tmp_path: Path) -> None:
    matching_path = _matching_checkpoint(tmp_path, StableFactory(), (2, 3, 4))

    manifest, trajectories = analyze_grid_stability(matching_path)

    assert manifest.grid_sizes == [2, 3, 4]
    assert len(trajectories) == 2
    assert all(item.level == StabilityLevel.STABLE for item in trajectories)
    assert all(item.grid_coverage == 1 for item in trajectories)
    assert all(item.transitions_survived == 2 for item in trajectories)
    assert all(item.minimum_jaccard == item.minimum_source_retention == 1 for item in trajectories)
    assert manifest.levels[StabilityLevel.STABLE] == 2
    assert all(summary.levels[StabilityLevel.STABLE] == 2 for summary in manifest.variants)


def test_grid_stability_keeps_ambiguous_and_unmatched_trajectories_explicit(tmp_path: Path) -> None:
    matching_path = _matching_checkpoint(tmp_path, SplitMergeFactory(), (2, 3))

    _, trajectories = analyze_grid_stability(matching_path)

    moderate = [item for item in trajectories if item.level == StabilityLevel.MODERATE]
    unmatched = [item for item in trajectories if item.level == StabilityLevel.UNMATCHED]
    assert len(moderate) == 1
    assert moderate[0].ambiguous_transition is True
    assert moderate[0].minimum_jaccard == 1 / 3
    assert len(unmatched) == 2
    assert all(item.transitions_survived == 0 for item in unmatched)


def test_grid_stability_resumes_and_detects_checkpoint_damage(tmp_path: Path) -> None:
    matching_path = _matching_checkpoint(tmp_path, StableFactory(), (2, 3))
    first_manifest, first = analyze_grid_stability(matching_path)

    resumed_manifest, resumed = analyze_grid_stability(matching_path)

    assert resumed_manifest == first_manifest
    assert resumed == first
    trajectories_path = matching_path.parent / "cluster-stability.jsonl"
    trajectories_path.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="incompatible or damaged"):
        analyze_grid_stability(matching_path)


def test_grid_stability_rejects_inverted_thresholds() -> None:
    with pytest.raises(ValueError, match="stable thresholds"):
        ClusterStabilityConfig(
            stable_minimum_jaccard=0.2,
            moderate_minimum_jaccard=0.3,
        )
