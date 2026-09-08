"""Resumable, source-bound HDBSCAN parameter-grid experiments."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from datetime import UTC, datetime
from enum import StrEnum
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.ml.clustering import (
    ClustererFactory,
    ClusteringManifest,
    ClusteringMetrics,
    HDBSCANConfig,
    cluster_corpus,
)

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

_MINIMUM_CLUSTER_SIZE = 2
ProgressCallback = Callable[[str], None]
NodeKey = tuple[int, int]


class _ExperimentModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ClusteringGridConfig(_ExperimentModel):
    """Define one controlled min-cluster-size experiment."""

    schema_version: int = 1
    min_cluster_sizes: tuple[int, ...] = (50, 100, 150, 250)
    base_config: HDBSCANConfig = Field(default_factory=HDBSCANConfig)

    @model_validator(mode="after")
    def validate_grid(self) -> ClusteringGridConfig:
        if not self.min_cluster_sizes:
            msg = "min_cluster_sizes cannot be empty"
            raise ValueError(msg)
        if any(type(value) is not int or value < _MINIMUM_CLUSTER_SIZE for value in self.min_cluster_sizes):
            msg = "min_cluster_sizes must contain integers greater than or equal to 2"
            raise ValueError(msg)
        if tuple(sorted(set(self.min_cluster_sizes))) != self.min_cluster_sizes:
            msg = "min_cluster_sizes must be unique and strictly increasing"
            raise ValueError(msg)
        return self


class ClusteringGridResult(_ExperimentModel):
    """Summarize one independently persisted HDBSCAN variant."""

    min_cluster_size: int = Field(ge=2)
    manifest_path: str
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    metrics: ClusteringMetrics
    warnings: list[str]


class ClusteringGridManifest(_ExperimentModel):
    """Bind every grid result to one reduction, corpus, and configuration."""

    schema_version: int = 1
    reduction_manifest_path: str
    reduction_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reduced_path: str
    reduced_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    corpus_manifest_path: str
    corpus_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    config: ClusteringGridConfig
    results: list[ClusteringGridResult]
    created_at: datetime

    @model_validator(mode="after")
    def validate_results(self) -> ClusteringGridManifest:
        if [result.min_cluster_size for result in self.results] != list(self.config.min_cluster_sizes):
            msg = "grid results do not match the configured min_cluster_sizes"
            raise ValueError(msg)
        return self


class ClusterTransitionStatus(StrEnum):
    """Describe a material cluster transition between adjacent grid variants."""

    MATCHED = "matched"
    SPLIT = "split"
    MERGED = "merged"
    SPLIT_AND_MERGED = "split_and_merged"
    DISAPPEARED = "disappeared"
    NEW = "new"


class ClusterTransition(_ExperimentModel):
    """Quantify one material overlap or an unmatched cluster."""

    source_min_cluster_size: int = Field(ge=2)
    target_min_cluster_size: int = Field(ge=2)
    source_cluster_id: int | None = Field(default=None, ge=0)
    target_cluster_id: int | None = Field(default=None, ge=0)
    source_records: int = Field(ge=0)
    target_records: int = Field(ge=0)
    overlap_records: int = Field(ge=0)
    jaccard: float = Field(ge=0, le=1)
    source_retention: float = Field(ge=0, le=1)
    target_composition: float = Field(ge=0, le=1)
    primary_match: bool
    status: ClusterTransitionStatus


class ClusterMatchingConfig(_ExperimentModel):
    """Control which overlaps are material enough to interpret."""

    schema_version: int = 1
    minimum_shared_records: int = Field(default=1, ge=1)
    minimum_overlap_share: float = Field(default=0.05, gt=0, le=1)


class ClusterMatchingManifest(_ExperimentModel):
    """Bind adjacent grid cluster transitions to verified label arrays."""

    schema_version: int = 1
    grid_manifest_path: str
    grid_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    config: ClusterMatchingConfig
    records: int = Field(ge=0)
    compared_pairs: list[tuple[int, int]]
    transitions_path: str
    transitions_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    transitions: int = Field(ge=0)
    created_at: datetime


class StabilityLevel(StrEnum):
    """Summarize robustness while preserving the underlying measurements."""

    STABLE = "stable"
    MODERATE = "moderate"
    FRAGILE = "fragile"
    UNMATCHED = "unmatched"


class ClusterStabilityConfig(_ExperimentModel):
    """Define explicit thresholds for grid-trajectory stability levels."""

    schema_version: int = 1
    stable_minimum_grid_coverage: float = Field(default=1.0, gt=0, le=1)
    stable_minimum_jaccard: float = Field(default=0.5, ge=0, le=1)
    stable_minimum_source_retention: float = Field(default=0.7, ge=0, le=1)
    moderate_minimum_grid_coverage: float = Field(default=0.5, gt=0, le=1)
    moderate_minimum_jaccard: float = Field(default=0.25, ge=0, le=1)
    moderate_minimum_source_retention: float = Field(default=0.5, ge=0, le=1)

    @model_validator(mode="after")
    def validate_threshold_order(self) -> ClusterStabilityConfig:
        pairs = (
            (self.stable_minimum_grid_coverage, self.moderate_minimum_grid_coverage),
            (self.stable_minimum_jaccard, self.moderate_minimum_jaccard),
            (self.stable_minimum_source_retention, self.moderate_minimum_source_retention),
        )
        if any(stable < moderate for stable, moderate in pairs):
            msg = "stable thresholds cannot be lower than moderate thresholds"
            raise ValueError(msg)
        return self


class ClusterTrajectoryNode(_ExperimentModel):
    """Identify one local cluster within a cross-variant trajectory."""

    min_cluster_size: int = Field(ge=2)
    cluster_id: int = Field(ge=0)
    records: int = Field(ge=1)


class ClusterStability(_ExperimentModel):
    """Describe one mutual-primary trajectory through the parameter grid."""

    trajectory_id: int = Field(ge=0)
    nodes: list[ClusterTrajectoryNode]
    variants_present: int = Field(ge=1)
    grid_coverage: float = Field(gt=0, le=1)
    transitions_survived: int = Field(ge=0)
    mean_jaccard: float | None = Field(default=None, ge=0, le=1)
    minimum_jaccard: float | None = Field(default=None, ge=0, le=1)
    mean_source_retention: float | None = Field(default=None, ge=0, le=1)
    minimum_source_retention: float | None = Field(default=None, ge=0, le=1)
    ambiguous_transition: bool
    level: StabilityLevel


class VariantStabilitySummary(_ExperimentModel):
    """Count stability levels among clusters in one grid variant."""

    min_cluster_size: int = Field(ge=2)
    clusters: int = Field(ge=0)
    levels: dict[StabilityLevel, int]


class ClusterStabilityManifest(_ExperimentModel):
    """Bind trajectory stability output to one verified matching checkpoint."""

    schema_version: int = 1
    matching_manifest_path: str
    matching_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    transitions_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    config: ClusterStabilityConfig
    grid_sizes: list[int]
    trajectories_path: str
    trajectories_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    trajectories: int = Field(ge=0)
    levels: dict[StabilityLevel, int]
    variants: list[VariantStabilitySummary]
    created_at: datetime


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_variant_artifacts(manifest: ClusteringManifest, variant_dir: Path) -> None:
    checks = (
        (manifest.labels_path, manifest.labels_sha256, "labels"),
        (manifest.probabilities_path, manifest.probabilities_sha256, "probabilities"),
        (manifest.summary_path, manifest.summary_sha256, "summary"),
        (manifest.model_path, manifest.model_sha256, "model"),
    )
    for raw_path, expected, label in checks:
        path = Path(raw_path).resolve()
        if not path.is_relative_to(variant_dir.resolve()):
            msg = f"grid variant {label} artifact escapes its directory"
            raise ValueError(msg)
        if not path.is_file() or _sha256_file(path) != expected:
            msg = f"grid variant {label} artifact is missing or has a checksum mismatch"
            raise ValueError(msg)


def _load_completed_variant(
    variant_dir: Path,
    expected_config: HDBSCANConfig,
    reduced_path: Path,
    reduction_manifest_path: Path,
    corpus_manifest_path: Path,
) -> ClusteringManifest | None:
    manifest_path = variant_dir / "clustering-manifest.json"
    if not manifest_path.exists():
        if variant_dir.exists() and any(variant_dir.iterdir()):
            msg = f"incomplete grid variant requires manual inspection: {variant_dir}"
            raise FileExistsError(msg)
        return None
    manifest = ClusteringManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    if manifest.config != expected_config:
        msg = f"completed grid variant has different configuration: {variant_dir.name}"
        raise ValueError(msg)
    if manifest.reduction_manifest_sha256 != _sha256_file(reduction_manifest_path):
        msg = f"completed grid variant uses a different reduction: {variant_dir.name}"
        raise ValueError(msg)
    if manifest.reduced_sha256 != _sha256_file(reduced_path):
        msg = f"completed grid variant uses a different reduced matrix: {variant_dir.name}"
        raise ValueError(msg)
    if manifest.corpus_manifest_sha256 != _sha256_file(corpus_manifest_path):
        msg = f"completed grid variant uses a different corpus: {variant_dir.name}"
        raise ValueError(msg)
    _validate_variant_artifacts(manifest, variant_dir)
    return manifest


def _result(manifest: ClusteringManifest, manifest_path: Path) -> ClusteringGridResult:
    return ClusteringGridResult(
        min_cluster_size=manifest.config.min_cluster_size,
        manifest_path=str(manifest_path),
        manifest_sha256=_sha256_file(manifest_path),
        metrics=manifest.metrics,
        warnings=manifest.warnings,
    )


def _validate_output_location(output_dir: Path, reduction_manifest_path: Path) -> None:
    source_run_dir = reduction_manifest_path.resolve().parent.parent
    if output_dir.resolve().is_relative_to(source_run_dir):
        msg = "clustering-grid output must be outside the immutable source run"
        raise ValueError(msg)


def run_clustering_grid(
    reduced_path: Path,
    reduction_manifest_path: Path,
    corpus_manifest_path: Path,
    output_dir: Path,
    *,
    config: ClusteringGridConfig | None = None,
    clusterer_factory: ClustererFactory | None = None,
    progress: ProgressCallback | None = None,
) -> ClusteringGridManifest:
    """Run or safely resume a min-cluster-size grid on one fixed UMAP matrix.

    Completed variants are checksum-validated and reused. A partial or incompatible
    variant blocks execution rather than being overwritten. The source run is never
    modified, and the aggregate manifest is published only after every variant is
    complete.
    """
    active_config = config or ClusteringGridConfig()
    for label, path in (
        ("reduced matrix", reduced_path),
        ("reduction manifest", reduction_manifest_path),
        ("corpus manifest", corpus_manifest_path),
    ):
        if not path.is_file():
            msg = f"{label} is missing: {path}"
            raise FileNotFoundError(msg)
    _validate_output_location(output_dir, reduction_manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_manifest_path = output_dir / "clustering-grid-manifest.json"
    source_hashes = (
        _sha256_file(reduction_manifest_path),
        _sha256_file(reduced_path),
        _sha256_file(corpus_manifest_path),
    )
    results = []
    for size in active_config.min_cluster_sizes:
        variant_dir = output_dir / f"min-cluster-size-{size}"
        variant_config = active_config.base_config.model_copy(update={"min_cluster_size": size})
        manifest = _load_completed_variant(
            variant_dir,
            variant_config,
            reduced_path,
            reduction_manifest_path,
            corpus_manifest_path,
        )
        if manifest is None:
            if progress is not None:
                progress(f"min_cluster_size={size}: fitting HDBSCAN")
            manifest = cluster_corpus(
                reduced_path,
                reduction_manifest_path,
                corpus_manifest_path,
                variant_dir,
                config=variant_config,
                clusterer_factory=clusterer_factory,
            )
            if progress is not None:
                progress(f"min_cluster_size={size}: completed and checkpointed")
        elif progress is not None:
            progress(f"min_cluster_size={size}: verified checkpoint reused")
        results.append(_result(manifest, variant_dir / "clustering-manifest.json"))
    grid_manifest = ClusteringGridManifest(
        reduction_manifest_path=str(reduction_manifest_path),
        reduction_manifest_sha256=source_hashes[0],
        reduced_path=str(reduced_path),
        reduced_sha256=source_hashes[1],
        corpus_manifest_path=str(corpus_manifest_path),
        corpus_manifest_sha256=source_hashes[2],
        config=active_config,
        results=results,
        created_at=datetime.now(UTC),
    )
    if final_manifest_path.exists():
        existing = ClusteringGridManifest.model_validate_json(final_manifest_path.read_text(encoding="utf-8"))
        comparable_fields = {"created_at"}
        if existing.model_dump(exclude=comparable_fields) != grid_manifest.model_dump(exclude=comparable_fields):
            msg = "existing clustering-grid manifest is incompatible with the requested experiment"
            raise ValueError(msg)
        return existing
    temporary = final_manifest_path.with_name(f".{final_manifest_path.name}.tmp")
    temporary.write_text(f"{grid_manifest.model_dump_json(indent=2)}\n", encoding="utf-8")
    temporary.replace(final_manifest_path)
    return grid_manifest


def _load_grid_labels(
    grid: ClusteringGridManifest,
    grid_dir: Path,
) -> dict[int, NDArray[np.int64]]:
    """Load aligned labels after validating the grid and every variant checksum."""
    import numpy as np

    for raw_path, expected, label in (
        (grid.reduction_manifest_path, grid.reduction_manifest_sha256, "reduction manifest"),
        (grid.reduced_path, grid.reduced_sha256, "reduced matrix"),
        (grid.corpus_manifest_path, grid.corpus_manifest_sha256, "corpus manifest"),
    ):
        path = Path(raw_path)
        if not path.is_file() or _sha256_file(path) != expected:
            msg = f"grid source {label} is missing or has a checksum mismatch"
            raise ValueError(msg)
    labels_by_size = {}
    expected_records = None
    for result in grid.results:
        manifest_path = Path(result.manifest_path).resolve()
        if not manifest_path.is_relative_to(grid_dir.resolve()):
            msg = "grid variant manifest escapes the grid directory"
            raise ValueError(msg)
        if not manifest_path.is_file() or _sha256_file(manifest_path) != result.manifest_sha256:
            msg = f"variant manifest checksum mismatch for min_cluster_size={result.min_cluster_size}"
            raise ValueError(msg)
        manifest = ClusteringManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        variant_dir = manifest_path.parent
        _validate_variant_artifacts(manifest, variant_dir)
        if (
            manifest.reduction_manifest_sha256 != grid.reduction_manifest_sha256
            or manifest.reduced_sha256 != grid.reduced_sha256
            or manifest.corpus_manifest_sha256 != grid.corpus_manifest_sha256
        ):
            msg = "variant manifest is not aligned with the grid sources"
            raise ValueError(msg)
        if manifest.config.min_cluster_size != result.min_cluster_size:
            msg = "variant manifest min_cluster_size disagrees with the grid"
            raise ValueError(msg)
        labels_path = Path(manifest.labels_path)
        labels = np.load(labels_path, mmap_mode="r", allow_pickle=False)
        if labels.shape != (manifest.output_records,) or labels.dtype.kind not in "iu":
            msg = f"invalid labels for min_cluster_size={result.min_cluster_size}"
            raise ValueError(msg)
        if manifest.output_records != manifest.input_records:
            msg = "cluster matching does not accept limited grid variants"
            raise ValueError(msg)
        if expected_records is None:
            expected_records = manifest.output_records
        elif manifest.output_records != expected_records:
            msg = "grid label arrays are not row-aligned"
            raise ValueError(msg)
        labels_by_size[result.min_cluster_size] = labels
    return labels_by_size


def _transition_status(source_degree: int, target_degree: int) -> ClusterTransitionStatus:
    if source_degree > 1 and target_degree > 1:
        return ClusterTransitionStatus.SPLIT_AND_MERGED
    if source_degree > 1:
        return ClusterTransitionStatus.SPLIT
    if target_degree > 1:
        return ClusterTransitionStatus.MERGED
    return ClusterTransitionStatus.MATCHED


def _match_label_pair(
    source: NDArray[np.int64],
    target: NDArray[np.int64],
    source_size: int,
    target_size: int,
    config: ClusterMatchingConfig,
) -> list[ClusterTransition]:
    """Match two aligned label arrays using material overlap and mutual primaries."""
    import numpy as np

    source_ids = sorted(int(value) for value in np.unique(source) if value >= 0)
    target_ids = sorted(int(value) for value in np.unique(target) if value >= 0)
    source_counts = {cluster_id: int(np.count_nonzero(source == cluster_id)) for cluster_id in source_ids}
    target_counts = {cluster_id: int(np.count_nonzero(target == cluster_id)) for cluster_id in target_ids}
    overlaps = np.zeros((len(source_ids), len(target_ids)), dtype=np.int64)
    source_positions = {cluster_id: index for index, cluster_id in enumerate(source_ids)}
    target_positions = {cluster_id: index for index, cluster_id in enumerate(target_ids)}
    clustered = (source >= 0) & (target >= 0)
    if np.any(clustered):
        pairs, counts = np.unique(
            np.column_stack((source[clustered], target[clustered])),
            axis=0,
            return_counts=True,
        )
        for (source_id, target_id), count in zip(pairs, counts, strict=True):
            overlaps[source_positions[int(source_id)], target_positions[int(target_id)]] = int(count)
    material = np.zeros_like(overlaps, dtype=bool)
    for left, source_id in enumerate(source_ids):
        for right, target_id in enumerate(target_ids):
            overlap = int(overlaps[left, right])
            material[left, right] = (
                overlap >= config.minimum_shared_records
                and overlap / source_counts[source_id] >= config.minimum_overlap_share
                and overlap / target_counts[target_id] >= config.minimum_overlap_share
            )
    primary_pairs: set[tuple[int, int]] = set()
    if source_ids and target_ids:
        source_choices = {
            left: min(
                (right for right in range(len(target_ids)) if material[left, right]),
                key=lambda right: (-int(overlaps[left, right]), target_ids[right]),
            )
            for left in range(len(source_ids))
            if material[left].any()
        }
        target_choices = {
            right: min(
                (left for left in range(len(source_ids)) if material[left, right]),
                key=lambda left: (-int(overlaps[left, right]), source_ids[left]),
            )
            for right in range(len(target_ids))
            if material[:, right].any()
        }
        primary_pairs = {
            (left, right)
            for left, right in source_choices.items()
            if target_choices.get(right) == left
        }
    source_degrees = material.sum(axis=1)
    target_degrees = material.sum(axis=0)
    transitions = []
    for left, source_id in enumerate(source_ids):
        for right, target_id in enumerate(target_ids):
            if not material[left, right]:
                continue
            overlap = int(overlaps[left, right])
            union = source_counts[source_id] + target_counts[target_id] - overlap
            transitions.append(
                ClusterTransition(
                    source_min_cluster_size=source_size,
                    target_min_cluster_size=target_size,
                    source_cluster_id=source_id,
                    target_cluster_id=target_id,
                    source_records=source_counts[source_id],
                    target_records=target_counts[target_id],
                    overlap_records=overlap,
                    jaccard=overlap / union,
                    source_retention=overlap / source_counts[source_id],
                    target_composition=overlap / target_counts[target_id],
                    primary_match=(left, right) in primary_pairs,
                    status=_transition_status(int(source_degrees[left]), int(target_degrees[right])),
                ),
            )
    for left, source_id in enumerate(source_ids):
        if source_degrees[left] == 0:
            transitions.append(
                ClusterTransition(
                    source_min_cluster_size=source_size,
                    target_min_cluster_size=target_size,
                    source_cluster_id=source_id,
                    target_cluster_id=None,
                    source_records=source_counts[source_id],
                    target_records=0,
                    overlap_records=0,
                    jaccard=0,
                    source_retention=0,
                    target_composition=0,
                    primary_match=False,
                    status=ClusterTransitionStatus.DISAPPEARED,
                ),
            )
    for right, target_id in enumerate(target_ids):
        if target_degrees[right] == 0:
            transitions.append(
                ClusterTransition(
                    source_min_cluster_size=source_size,
                    target_min_cluster_size=target_size,
                    source_cluster_id=None,
                    target_cluster_id=target_id,
                    source_records=0,
                    target_records=target_counts[target_id],
                    overlap_records=0,
                    jaccard=0,
                    source_retention=0,
                    target_composition=0,
                    primary_match=False,
                    status=ClusterTransitionStatus.NEW,
                ),
            )
    return sorted(
        transitions,
        key=lambda item: (
            item.source_cluster_id if item.source_cluster_id is not None else -1,
            item.target_cluster_id if item.target_cluster_id is not None else -1,
        ),
    )


def match_grid_clusters(
    grid_manifest_path: Path,
    *,
    config: ClusterMatchingConfig | None = None,
) -> tuple[ClusterMatchingManifest, tuple[ClusterTransition, ...]]:
    """Match clusters across adjacent, row-aligned grid variants and checkpoint the result."""
    active_config = config or ClusterMatchingConfig()
    grid_dir = grid_manifest_path.resolve().parent
    if not grid_manifest_path.is_file():
        msg = f"clustering-grid manifest is missing: {grid_manifest_path}"
        raise FileNotFoundError(msg)
    grid = ClusteringGridManifest.model_validate_json(grid_manifest_path.read_text(encoding="utf-8"))
    labels_by_size = _load_grid_labels(grid, grid_dir)
    pairs = list(pairwise(grid.config.min_cluster_sizes))
    transitions = tuple(
        transition
        for source_size, target_size in pairs
        for transition in _match_label_pair(
            labels_by_size[source_size],
            labels_by_size[target_size],
            source_size,
            target_size,
            active_config,
        )
    )
    transitions_path = grid_dir / "cluster-transitions.jsonl"
    manifest_path = grid_dir / "cluster-matching-manifest.json"
    if transitions_path.exists() != manifest_path.exists():
        msg = "incomplete cluster-matching checkpoint requires manual inspection"
        raise FileExistsError(msg)
    if manifest_path.exists():
        existing = ClusterMatchingManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        if (
            existing.grid_manifest_sha256 != _sha256_file(grid_manifest_path)
            or existing.config != active_config
            or _sha256_file(transitions_path) != existing.transitions_sha256
        ):
            msg = "existing cluster-matching checkpoint is incompatible or damaged"
            raise ValueError(msg)
        persisted = tuple(
            ClusterTransition.model_validate_json(line)
            for line in transitions_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if persisted != transitions:
            msg = "persisted cluster transitions disagree with verified grid labels"
            raise ValueError(msg)
        return existing, persisted
    transitions_tmp = transitions_path.with_name(f".{transitions_path.name}.tmp")
    manifest_tmp = manifest_path.with_name(f".{manifest_path.name}.tmp")
    with transitions_tmp.open("w", encoding="utf-8") as target_file:
        for transition in transitions:
            target_file.write(f"{transition.model_dump_json()}\n")
    transitions_tmp.replace(transitions_path)
    manifest = ClusterMatchingManifest(
        grid_manifest_path=str(grid_manifest_path),
        grid_manifest_sha256=_sha256_file(grid_manifest_path),
        config=active_config,
        records=len(next(iter(labels_by_size.values()), ())),
        compared_pairs=pairs,
        transitions_path=str(transitions_path),
        transitions_sha256=_sha256_file(transitions_path),
        transitions=len(transitions),
        created_at=datetime.now(UTC),
    )
    manifest_tmp.write_text(f"{manifest.model_dump_json(indent=2)}\n", encoding="utf-8")
    manifest_tmp.replace(manifest_path)
    return manifest, transitions


def _load_matching_checkpoint(
    matching_manifest_path: Path,
) -> tuple[ClusterMatchingManifest, tuple[ClusterTransition, ...], list[int]]:
    """Load a checksum-verified matching checkpoint and its grid sizes."""
    if not matching_manifest_path.is_file():
        msg = f"cluster-matching manifest is missing: {matching_manifest_path}"
        raise FileNotFoundError(msg)
    matching = ClusterMatchingManifest.model_validate_json(
        matching_manifest_path.read_text(encoding="utf-8"),
    )
    transitions_path = Path(matching.transitions_path).resolve()
    if not transitions_path.is_relative_to(matching_manifest_path.resolve().parent):
        msg = "cluster transitions escape the matching checkpoint directory"
        raise ValueError(msg)
    if not transitions_path.is_file() or _sha256_file(transitions_path) != matching.transitions_sha256:
        msg = "cluster transitions are missing or have a checksum mismatch"
        raise ValueError(msg)
    transitions = tuple(
        ClusterTransition.model_validate_json(line)
        for line in transitions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if len(transitions) != matching.transitions:
        msg = "cluster transition count disagrees with its manifest"
        raise ValueError(msg)
    grid_path = Path(matching.grid_manifest_path)
    if not grid_path.is_file() or _sha256_file(grid_path) != matching.grid_manifest_sha256:
        msg = "matching checkpoint grid manifest is missing or has a checksum mismatch"
        raise ValueError(msg)
    grid = ClusteringGridManifest.model_validate_json(grid_path.read_text(encoding="utf-8"))
    grid_sizes = list(grid.config.min_cluster_sizes)
    expected_pairs = list(pairwise(grid_sizes))
    if len(grid_sizes) < _MINIMUM_CLUSTER_SIZE or matching.compared_pairs != expected_pairs:
        msg = "matching checkpoint does not cover adjacent grid variants"
        raise ValueError(msg)
    if any(
        (item.source_min_cluster_size, item.target_min_cluster_size) not in expected_pairs
        for item in transitions
    ):
        msg = "cluster transition refers to a non-adjacent grid pair"
        raise ValueError(msg)
    return matching, transitions, grid_sizes


def _trajectory_level(
    coverage: float,
    minimum_jaccard: float | None,
    minimum_retention: float | None,
    ambiguous: bool,
    transitions: int,
    config: ClusterStabilityConfig,
) -> StabilityLevel:
    if transitions == 0 or minimum_jaccard is None or minimum_retention is None:
        return StabilityLevel.UNMATCHED
    if (
        not ambiguous
        and coverage >= config.stable_minimum_grid_coverage
        and minimum_jaccard >= config.stable_minimum_jaccard
        and minimum_retention >= config.stable_minimum_source_retention
    ):
        return StabilityLevel.STABLE
    if (
        coverage >= config.moderate_minimum_grid_coverage
        and minimum_jaccard >= config.moderate_minimum_jaccard
        and minimum_retention >= config.moderate_minimum_source_retention
    ):
        return StabilityLevel.MODERATE
    return StabilityLevel.FRAGILE


def _build_cluster_trajectories(
    transitions: tuple[ClusterTransition, ...],
    grid_sizes: list[int],
    config: ClusterStabilityConfig,
) -> tuple[ClusterStability, ...]:
    """Build non-branching trajectories from mutual-primary transition edges."""
    records: dict[NodeKey, int] = {}
    primary_out: dict[NodeKey, tuple[NodeKey, ClusterTransition]] = {}
    primary_in: dict[NodeKey, NodeKey] = {}
    for item in transitions:
        source_key = (
            (item.source_min_cluster_size, item.source_cluster_id)
            if item.source_cluster_id is not None
            else None
        )
        target_key = (
            (item.target_min_cluster_size, item.target_cluster_id)
            if item.target_cluster_id is not None
            else None
        )
        for key, count in ((source_key, item.source_records), (target_key, item.target_records)):
            if key is None:
                continue
            if key in records and records[key] != count:
                msg = f"cluster {key} has inconsistent sizes across transitions"
                raise ValueError(msg)
            records[key] = count
        if not item.primary_match:
            continue
        if source_key is None or target_key is None:
            msg = "primary cluster transition must have source and target clusters"
            raise ValueError(msg)
        if source_key in primary_out or target_key in primary_in:
            msg = "primary cluster transitions are not one-to-one"
            raise ValueError(msg)
        primary_out[source_key] = (target_key, item)
        primary_in[target_key] = source_key
    roots = sorted(key for key in records if key not in primary_in)
    visited: set[NodeKey] = set()
    trajectories: list[ClusterStability] = []
    for root in roots:
        node_keys = [root]
        edges = []
        current = root
        while current in primary_out:
            target, edge = primary_out[current]
            if target in node_keys:
                msg = "cluster trajectory contains a cycle"
                raise ValueError(msg)
            edges.append(edge)
            node_keys.append(target)
            current = target
        visited.update(node_keys)
        jaccards = [edge.jaccard for edge in edges]
        retentions = [edge.source_retention for edge in edges]
        coverage = len(node_keys) / len(grid_sizes)
        minimum_jaccard = min(jaccards, default=None)
        minimum_retention = min(retentions, default=None)
        ambiguous = any(edge.status != ClusterTransitionStatus.MATCHED for edge in edges)
        trajectories.append(
            ClusterStability(
                trajectory_id=len(trajectories),
                nodes=[
                    ClusterTrajectoryNode(
                        min_cluster_size=size,
                        cluster_id=cluster_id,
                        records=records[(size, cluster_id)],
                    )
                    for size, cluster_id in node_keys
                ],
                variants_present=len(node_keys),
                grid_coverage=coverage,
                transitions_survived=len(edges),
                mean_jaccard=sum(jaccards) / len(jaccards) if jaccards else None,
                minimum_jaccard=minimum_jaccard,
                mean_source_retention=sum(retentions) / len(retentions) if retentions else None,
                minimum_source_retention=minimum_retention,
                ambiguous_transition=ambiguous,
                level=_trajectory_level(
                    coverage,
                    minimum_jaccard,
                    minimum_retention,
                    ambiguous,
                    len(edges),
                    config,
                ),
            ),
        )
    if visited != records.keys():
        msg = "not every matched cluster belongs to a stability trajectory"
        raise ValueError(msg)
    return tuple(trajectories)


def _variant_stability_summaries(
    trajectories: tuple[ClusterStability, ...],
    grid_sizes: list[int],
) -> list[VariantStabilitySummary]:
    summaries = []
    for size in grid_sizes:
        levels = dict.fromkeys(StabilityLevel, 0)
        clusters = 0
        for trajectory in trajectories:
            if any(node.min_cluster_size == size for node in trajectory.nodes):
                levels[trajectory.level] += 1
                clusters += 1
        summaries.append(VariantStabilitySummary(min_cluster_size=size, clusters=clusters, levels=levels))
    return summaries


def analyze_grid_stability(
    matching_manifest_path: Path,
    *,
    config: ClusterStabilityConfig | None = None,
) -> tuple[ClusterStabilityManifest, tuple[ClusterStability, ...]]:
    """Measure and checkpoint transparent stability trajectories across a grid."""
    active_config = config or ClusterStabilityConfig()
    matching, transitions, grid_sizes = _load_matching_checkpoint(matching_manifest_path)
    trajectories = _build_cluster_trajectories(transitions, grid_sizes, active_config)
    output_dir = matching_manifest_path.resolve().parent
    trajectories_path = output_dir / "cluster-stability.jsonl"
    manifest_path = output_dir / "cluster-stability-manifest.json"
    if trajectories_path.exists() != manifest_path.exists():
        msg = "incomplete cluster-stability checkpoint requires manual inspection"
        raise FileExistsError(msg)
    if manifest_path.exists():
        existing = ClusterStabilityManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        if (
            existing.matching_manifest_sha256 != _sha256_file(matching_manifest_path)
            or existing.transitions_sha256 != matching.transitions_sha256
            or existing.config != active_config
            or _sha256_file(trajectories_path) != existing.trajectories_sha256
        ):
            msg = "existing cluster-stability checkpoint is incompatible or damaged"
            raise ValueError(msg)
        persisted = tuple(
            ClusterStability.model_validate_json(line)
            for line in trajectories_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if persisted != trajectories:
            msg = "persisted stability trajectories disagree with verified transitions"
            raise ValueError(msg)
        return existing, persisted
    trajectories_tmp = trajectories_path.with_name(f".{trajectories_path.name}.tmp")
    manifest_tmp = manifest_path.with_name(f".{manifest_path.name}.tmp")
    with trajectories_tmp.open("w", encoding="utf-8") as target_file:
        for trajectory in trajectories:
            target_file.write(f"{trajectory.model_dump_json()}\n")
    trajectories_tmp.replace(trajectories_path)
    levels = dict.fromkeys(StabilityLevel, 0)
    for trajectory in trajectories:
        levels[trajectory.level] += 1
    manifest = ClusterStabilityManifest(
        matching_manifest_path=str(matching_manifest_path),
        matching_manifest_sha256=_sha256_file(matching_manifest_path),
        transitions_sha256=matching.transitions_sha256,
        config=active_config,
        grid_sizes=grid_sizes,
        trajectories_path=str(trajectories_path),
        trajectories_sha256=_sha256_file(trajectories_path),
        trajectories=len(trajectories),
        levels=levels,
        variants=_variant_stability_summaries(trajectories, grid_sizes),
        created_at=datetime.now(UTC),
    )
    manifest_tmp.write_text(f"{manifest.model_dump_json(indent=2)}\n", encoding="utf-8")
    manifest_tmp.replace(manifest_path)
    return manifest, trajectories
