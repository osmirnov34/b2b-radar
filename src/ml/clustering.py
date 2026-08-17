"""Checksum-verified HDBSCAN clustering for the stage 8 clustering space."""

from __future__ import annotations

import hashlib
import pickle
from collections import Counter, defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.ml.corpus import CorpusManifest, CorpusRecord
from src.ml.dimensionality_reduction import ReductionArtifactManifest, ReductionMode
from src.ml.schemas import TextKind

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

_EXPECTED_MATRIX_DIMENSIONS = 2
_MINIMUM_CLUSTERING_RECORDS = 2


class _ClusteringModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class HDBSCANConfig(_ClusteringModel):
    schema_version: int = 1
    dataset_role: Literal["development"] = "development"
    min_cluster_size: int = Field(default=250, ge=2)
    min_samples: int | None = Field(default=None, ge=1)
    metric: str = Field(default="euclidean", min_length=1)
    cluster_selection_method: Literal["eom", "leaf"] = "eom"
    cluster_selection_epsilon: float = Field(default=0.0, ge=0)
    alpha: float = Field(default=1.0, gt=0)
    allow_single_cluster: bool = False
    prediction_data: bool = True
    threads: int = Field(default=1, ge=1)
    minimum_probability: float = Field(default=0.5, ge=0, le=1)
    dbcv_sample_size: int = Field(default=0, ge=0)
    max_outlier_share_warning: float = Field(default=0.7, ge=0, le=1)
    max_dominant_cluster_share_warning: float = Field(default=0.5, ge=0, le=1)
    max_micro_cluster_share_warning: float = Field(default=0.5, ge=0, le=1)
    min_mean_probability_warning: float = Field(default=0.5, ge=0, le=1)
    random_seed: int = 42


class ClusterSummary(_ClusteringModel):
    cluster_id: int = Field(ge=0)
    records: int = Field(ge=1)
    corpus_share: float = Field(gt=0, le=1)
    mean_probability: float = Field(ge=0, le=1)
    median_probability: float = Field(ge=0, le=1)
    minimum_probability: float = Field(ge=0, le=1)
    comments: int = Field(ge=0)
    replies: int = Field(ge=0)
    languages: dict[str, int]
    unique_videos: int = Field(ge=0)
    minimum_record_index: int = Field(ge=0)


class ClusteringMetrics(_ClusteringModel):
    records: int = Field(ge=0)
    clusters: int = Field(ge=0)
    outliers: int = Field(ge=0)
    outlier_share: float = Field(ge=0, le=1)
    smallest_cluster: int | None = Field(default=None, ge=1)
    largest_cluster: int | None = Field(default=None, ge=1)
    median_cluster_size: float | None = Field(default=None, ge=1)
    dominant_cluster_share: float = Field(ge=0, le=1)
    micro_clusters: int = Field(ge=0)
    micro_cluster_share: float = Field(ge=0, le=1)
    mean_probability: float = Field(ge=0, le=1)
    low_confidence_records: int = Field(ge=0)
    low_confidence_share: float = Field(ge=0, le=1)
    relative_validity: float | None = Field(default=None, ge=-1, le=1)
    dbcv: float | None = Field(default=None, ge=-1, le=1)


class ClusteringManifest(_ClusteringModel):
    schema_version: int = 1
    dataset_role: Literal["development"] = "development"
    reduction_manifest_path: str
    reduction_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reduced_path: str
    reduced_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    corpus_manifest_path: str
    corpus_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_records: int = Field(ge=0)
    output_records: int = Field(ge=0)
    input_dimensions: int = Field(ge=2)
    config: HDBSCANConfig
    label_mapping: dict[int, int]
    labels_path: str
    labels_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    probabilities_path: str
    probabilities_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    summary_path: str
    summary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_path: str
    model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    library: str
    library_version: str
    metrics: ClusteringMetrics
    warnings: list[str]
    created_at: datetime

    @model_validator(mode="after")
    def validate_counts(self) -> ClusteringManifest:
        if self.output_records > self.input_records:
            msg = "clustering output cannot contain more rows than its reduction input"
            raise ValueError(msg)
        return self


class Clusterer(Protocol):
    """Injectable clustering boundary used by production and lightweight tests."""

    library: str
    library_version: str

    @property
    def probabilities(self) -> NDArray[np.floating]: ...

    @property
    def relative_validity(self) -> float | None: ...

    @property
    def dbcv(self) -> float | None: ...

    def fit_predict(self, vectors: NDArray[np.float32]) -> NDArray[np.integer]: ...

    def dump(self, path: Path) -> None: ...


ClustererFactory = Callable[[HDBSCANConfig], Clusterer]


class HDBSCANClusterer:
    """Production hdbscan adapter; model pickle is written but never loaded."""

    library = "hdbscan"

    def __init__(self, config: HDBSCANConfig) -> None:
        try:
            import hdbscan
        except ImportError as exc:  # pragma: no cover - optional ML dependency
            msg = "hdbscan is required; install the 'analysis' extra"
            raise RuntimeError(msg) from exc
        self.library_version = version("hdbscan")
        self._config = config
        self._model = hdbscan.HDBSCAN(
            min_cluster_size=config.min_cluster_size,
            min_samples=config.min_samples,
            metric=config.metric,
            cluster_selection_method=config.cluster_selection_method,
            cluster_selection_epsilon=config.cluster_selection_epsilon,
            alpha=config.alpha,
            allow_single_cluster=config.allow_single_cluster,
            prediction_data=config.prediction_data,
            core_dist_n_jobs=config.threads,
            gen_min_span_tree=True,
        )
        self._dbcv: float | None = None

    @property
    def probabilities(self) -> NDArray[np.floating]:
        import numpy as np

        return np.asarray(self._model.probabilities_, dtype=np.float32)

    @property
    def relative_validity(self) -> float | None:
        value = getattr(self._model, "relative_validity_", None)
        return None if value is None else float(value)

    @property
    def dbcv(self) -> float | None:
        return self._dbcv

    def fit_predict(self, vectors: NDArray[np.float32]) -> NDArray[np.integer]:
        import numpy as np

        labels = np.asarray(self._model.fit_predict(vectors), dtype=np.int64)
        sample_size = min(self._config.dbcv_sample_size, len(vectors))
        generator = np.random.default_rng(self._config.random_seed)
        sample_indices = np.sort(generator.choice(len(vectors), size=sample_size, replace=False))
        sample_labels = labels[sample_indices]
        clusters = {int(label) for label in sample_labels if label >= 0}
        if sample_size and len(clusters) >= _MINIMUM_CLUSTERING_RECORDS:
            from hdbscan.validity import validity_index

            self._dbcv = float(
                validity_index(
                    np.asarray(vectors[sample_indices], dtype=np.float64),
                    sample_labels,
                    metric=self._config.metric,
                ),
            )
        return labels

    def dump(self, path: Path) -> None:
        with path.open("wb") as target:
            pickle.dump(self._model, target, protocol=pickle.HIGHEST_PROTOCOL)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_contracts(
    reduced_path: Path,
    reduction_manifest_path: Path,
    corpus_manifest_path: Path,
) -> tuple[ReductionArtifactManifest, CorpusManifest]:
    reduction = ReductionArtifactManifest.model_validate_json(reduction_manifest_path.read_text(encoding="utf-8"))
    corpus = CorpusManifest.model_validate_json(corpus_manifest_path.read_text(encoding="utf-8"))
    if reduction.mode != ReductionMode.CLUSTERING:
        msg = f"HDBSCAN requires a clustering reduction manifest, got {reduction.mode.value!r}"
        raise ValueError(msg)
    if _sha256_file(reduced_path) != reduction.reduced_sha256:
        msg = "reduced matrix checksum does not match reduction manifest"
        raise ValueError(msg)
    if _sha256_file(corpus_manifest_path) != reduction.corpus_manifest_sha256:
        msg = "corpus manifest checksum does not match reduction manifest"
        raise ValueError(msg)
    corpus_path = Path(corpus.corpus_path)
    ids_path = Path(corpus.final_record_ids_path)
    if not corpus_path.is_file() or _sha256_file(corpus_path) != corpus.corpus_sha256:
        msg = "final corpus is missing or does not match corpus manifest"
        raise ValueError(msg)
    if not ids_path.is_file() or _sha256_file(ids_path) != corpus.final_record_ids_sha256:
        msg = "final record IDs are missing or do not match corpus manifest"
        raise ValueError(msg)
    if reduction.input_records != corpus.stats.output_records:
        msg = "reduction and corpus manifests disagree on input record count"
        raise ValueError(msg)
    return reduction, corpus


def _validate_matrix(vectors: NDArray[np.floating], expected_shape: tuple[int, int]) -> NDArray[np.float32]:
    import numpy as np

    if vectors.ndim != _EXPECTED_MATRIX_DIMENSIONS or vectors.shape != expected_shape:
        msg = f"clustering matrix shape {vectors.shape} does not match expected {expected_shape}"
        raise ValueError(msg)
    if vectors.dtype != np.float32:
        msg = f"clustering matrix dtype must be float32, got {vectors.dtype}"
        raise ValueError(msg)
    result = np.asarray(vectors, dtype=np.float32)
    if not np.isfinite(result).all():
        msg = "clustering matrix contains NaN or infinite values"
        raise ValueError(msg)
    return result


def normalize_cluster_labels(labels: NDArray[np.integer]) -> tuple[NDArray[np.int64], dict[int, int]]:
    """Map clusters by descending size and earliest row while preserving outlier -1."""
    import numpy as np

    values = np.asarray(labels)
    if values.ndim != 1 or values.dtype.kind not in "iu":
        msg = "cluster labels must be a one-dimensional integer array"
        raise ValueError(msg)
    if np.any(values < -1):
        msg = "cluster labels may only use -1 or non-negative integers"
        raise ValueError(msg)
    groups: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(values):
        if label >= 0:
            groups[int(label)].append(index)
    ordered = sorted(groups, key=lambda label: (-len(groups[label]), groups[label][0], label))
    mapping = {original: normalized for normalized, original in enumerate(ordered)}
    normalized = np.asarray([mapping.get(int(label), -1) for label in values], dtype=np.int64)
    return normalized, mapping


def _validate_clusterer_result(
    raw_labels: NDArray[np.integer],
    raw_probabilities: NDArray[np.floating],
    expected_records: int,
) -> tuple[NDArray[np.int64], NDArray[np.float32], dict[int, int]]:
    import numpy as np

    labels = np.asarray(raw_labels)
    probabilities = np.asarray(raw_probabilities, dtype=np.float32)
    if labels.shape != (expected_records,):
        msg = f"clusterer returned labels shape {labels.shape}; expected {(expected_records,)}"
        raise ValueError(msg)
    if probabilities.shape != (expected_records,):
        msg = f"clusterer returned probabilities shape {probabilities.shape}; expected {(expected_records,)}"
        raise ValueError(msg)
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        msg = "cluster probabilities must be finite values in [0, 1]"
        raise ValueError(msg)
    normalized, mapping = normalize_cluster_labels(labels)
    return normalized, probabilities, mapping


class _SummaryState:
    def __init__(self) -> None:
        self.indices: list[int] = []
        self.probabilities: list[float] = []
        self.kinds: Counter[TextKind] = Counter()
        self.languages: Counter[str] = Counter()
        self.videos: set[str] = set()


def _build_summaries(
    corpus_path: Path,
    labels: NDArray[np.int64],
    probabilities: NDArray[np.float32],
) -> list[ClusterSummary]:
    states: dict[int, _SummaryState] = defaultdict(_SummaryState)
    count = 0
    with corpus_path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            if count >= len(labels):
                break
            record = CorpusRecord.model_validate_json(line)
            cluster_id = int(labels[count])
            if cluster_id >= 0:
                state = states[cluster_id]
                state.indices.append(count)
                state.probabilities.append(float(probabilities[count]))
                state.kinds[record.text_kind] += 1
                state.languages[record.detected_language] += 1
                if record.video_id:
                    state.videos.add(record.video_id)
            count += 1
    if count != len(labels):
        msg = f"final corpus supplied {count} records for {len(labels)} cluster labels"
        raise ValueError(msg)
    summaries = []
    for cluster_id in sorted(states):
        state = states[cluster_id]
        summaries.append(
            ClusterSummary(
                cluster_id=cluster_id,
                records=len(state.indices),
                corpus_share=len(state.indices) / len(labels),
                mean_probability=sum(state.probabilities) / len(state.probabilities),
                median_probability=float(median(state.probabilities)),
                minimum_probability=min(state.probabilities),
                comments=state.kinds[TextKind.COMMENT],
                replies=state.kinds[TextKind.REPLY],
                languages=dict(state.languages),
                unique_videos=len(state.videos),
                minimum_record_index=state.indices[0],
            ),
        )
    return summaries


def _metrics(
    labels: NDArray[np.int64],
    probabilities: NDArray[np.float32],
    summaries: list[ClusterSummary],
    config: HDBSCANConfig,
    clusterer: Clusterer,
) -> ClusteringMetrics:
    import numpy as np

    records = len(labels)
    outliers = int(np.count_nonzero(labels == -1))
    sizes = [summary.records for summary in summaries]
    micro_clusters = sum(size < 2 * config.min_cluster_size for size in sizes)
    low_confidence = int(np.count_nonzero(probabilities < config.minimum_probability))
    return ClusteringMetrics(
        records=records,
        clusters=len(summaries),
        outliers=outliers,
        outlier_share=outliers / records,
        smallest_cluster=min(sizes, default=None),
        largest_cluster=max(sizes, default=None),
        median_cluster_size=float(median(sizes)) if sizes else None,
        dominant_cluster_share=max(sizes, default=0) / records,
        micro_clusters=micro_clusters,
        micro_cluster_share=micro_clusters / len(sizes) if sizes else 0,
        mean_probability=float(np.mean(probabilities)),
        low_confidence_records=low_confidence,
        low_confidence_share=low_confidence / records,
        relative_validity=clusterer.relative_validity,
        dbcv=clusterer.dbcv,
    )


def _warnings(metrics: ClusteringMetrics, config: HDBSCANConfig) -> list[str]:
    warnings = []
    if metrics.outlier_share > config.max_outlier_share_warning:
        warnings.append(f"high outlier share: {metrics.outlier_share:.1%}")
    if metrics.dominant_cluster_share > config.max_dominant_cluster_share_warning:
        warnings.append(f"dominant cluster contains {metrics.dominant_cluster_share:.1%} of records")
    if metrics.mean_probability < config.min_mean_probability_warning:
        warnings.append(f"low mean membership probability: {metrics.mean_probability:.3f}")
    if metrics.micro_cluster_share > config.max_micro_cluster_share_warning:
        warnings.append(f"high micro-cluster share: {metrics.micro_cluster_share:.1%}")
    if metrics.clusters == 0:
        warnings.append("HDBSCAN produced only outliers")
    return warnings


def _save_numpy(path: Path, values: NDArray[np.generic]) -> None:
    import numpy as np

    with path.open("wb") as target:
        np.save(target, values, allow_pickle=False)


def _ensure_reduction_unchanged(path: Path, expected_sha256: str) -> None:
    if _sha256_file(path) != expected_sha256:
        msg = "clustering matrix changed while HDBSCAN was running"
        raise ValueError(msg)


def cluster_corpus(
    reduced_path: Path,
    reduction_manifest_path: Path,
    corpus_manifest_path: Path,
    output_dir: Path,
    *,
    config: HDBSCANConfig | None = None,
    clusterer_factory: ClustererFactory | None = None,
    force: bool = False,
    limit: int | None = None,
) -> ClusteringManifest:
    """Fit HDBSCAN and publish normalized, corpus-aligned clustering artifacts."""
    import numpy as np

    active_config = config or HDBSCANConfig()
    reduction, corpus = _validate_contracts(reduced_path, reduction_manifest_path, corpus_manifest_path)
    source = np.load(reduced_path, mmap_mode="r", allow_pickle=False)
    _validate_matrix(source, (reduction.output_records, reduction.output_dimensions))
    output_records = reduction.output_records if limit is None else min(limit, reduction.output_records)
    if output_records < _MINIMUM_CLUSTERING_RECORDS:
        msg = f"HDBSCAN requires at least {_MINIMUM_CLUSTERING_RECORDS} records"
        raise ValueError(msg)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = output_dir / "cluster-labels.npy"
    probabilities_path = output_dir / "cluster-probabilities.npy"
    summary_path = output_dir / "cluster-summary.jsonl"
    model_path = output_dir / "hdbscan-model.pkl"
    manifest_path = output_dir / "clustering-manifest.json"
    report_path = output_dir / "clustering-report.md"
    finals = (labels_path, probabilities_path, summary_path, model_path, manifest_path, report_path)
    existing = next((path for path in finals if path.exists()), None)
    if existing is not None and not force:
        msg = f"refusing to overwrite existing clustering artifact: {existing}"
        raise FileExistsError(msg)
    vectors = np.asarray(source[:output_records], dtype=np.float32)
    factory = clusterer_factory or HDBSCANClusterer
    clusterer = factory(active_config)
    raw_labels = clusterer.fit_predict(vectors)
    labels, probabilities, mapping = _validate_clusterer_result(
        raw_labels,
        clusterer.probabilities,
        output_records,
    )
    summaries = _build_summaries(Path(corpus.corpus_path), labels, probabilities)
    metrics = _metrics(labels, probabilities, summaries, active_config, clusterer)
    warnings = _warnings(metrics, active_config)
    temporary = {path: path.with_name(f".{path.name}.tmp") for path in finals}
    for path in temporary.values():
        path.unlink(missing_ok=True)
    try:
        _save_numpy(temporary[labels_path], labels)
        _save_numpy(temporary[probabilities_path], probabilities)
        with temporary[summary_path].open("w", encoding="utf-8") as target:
            for summary in summaries:
                target.write(f"{summary.model_dump_json()}\n")
        clusterer.dump(temporary[model_path])
        _ensure_reduction_unchanged(reduced_path, reduction.reduced_sha256)
        manifest = ClusteringManifest(
            reduction_manifest_path=str(reduction_manifest_path),
            reduction_manifest_sha256=_sha256_file(reduction_manifest_path),
            reduced_path=str(reduced_path),
            reduced_sha256=reduction.reduced_sha256,
            corpus_manifest_path=str(corpus_manifest_path),
            corpus_manifest_sha256=_sha256_file(corpus_manifest_path),
            corpus_sha256=corpus.corpus_sha256,
            input_records=reduction.output_records,
            output_records=output_records,
            input_dimensions=reduction.output_dimensions,
            config=active_config,
            label_mapping=mapping,
            labels_path=str(labels_path),
            labels_sha256=_sha256_file(temporary[labels_path]),
            probabilities_path=str(probabilities_path),
            probabilities_sha256=_sha256_file(temporary[probabilities_path]),
            summary_path=str(summary_path),
            summary_sha256=_sha256_file(temporary[summary_path]),
            model_path=str(model_path),
            model_sha256=_sha256_file(temporary[model_path]),
            library=clusterer.library,
            library_version=clusterer.library_version,
            metrics=metrics,
            warnings=warnings,
            created_at=datetime.now(UTC),
        )
        temporary[manifest_path].write_text(f"{manifest.model_dump_json(indent=2)}\n", encoding="utf-8")
        temporary[report_path].write_text(_clustering_report(manifest), encoding="utf-8")
        for final in (labels_path, probabilities_path, summary_path, model_path, report_path, manifest_path):
            temporary[final].replace(final)
    except Exception:
        for path in temporary.values():
            path.unlink(missing_ok=True)
        raise
    return manifest


def _clustering_report(manifest: ClusteringManifest) -> str:
    metrics = manifest.metrics
    warnings = "\n".join(f"- {warning}" for warning in manifest.warnings) or "- None."
    return f"""# HDBSCAN clustering

- Dataset role: `{manifest.dataset_role}`
- Records: {metrics.records}
- Clusters: {metrics.clusters}
- Outliers: {metrics.outliers} ({metrics.outlier_share:.1%})
- Cluster size min/median/max: {metrics.smallest_cluster}/{metrics.median_cluster_size}/{metrics.largest_cluster}
- Dominant cluster share: {metrics.dominant_cluster_share:.1%}
- Micro-clusters: {metrics.micro_clusters} ({metrics.micro_cluster_share:.1%})
- Mean membership probability: {metrics.mean_probability:.4f}
- Low-confidence share: {metrics.low_confidence_share:.1%}
- Relative validity: {metrics.relative_validity}
- DBCV: {metrics.dbcv}
- Labels SHA-256: `{manifest.labels_sha256}`
- Probabilities SHA-256: `{manifest.probabilities_sha256}`
- Model: `{manifest.library}=={manifest.library_version}`

## Warnings

{warnings}

The model pickle is a local trusted artifact and must not be loaded without checksum and origin verification. This
report and the cluster summary contain aggregate values only, without comment text or source identifiers.
"""
