"""Resumable, source-bound HDBSCAN parameter-grid experiments."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.ml.clustering import (
    ClustererFactory,
    ClusteringManifest,
    ClusteringMetrics,
    HDBSCANConfig,
    cluster_corpus,
)

_MINIMUM_CLUSTER_SIZE = 2
ProgressCallback = Callable[[str], None]


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
