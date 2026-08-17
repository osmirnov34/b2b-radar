"""Reproducible UMAP reduction for clustering and visualization spaces."""

from __future__ import annotations

import hashlib
import json
import pickle
from collections.abc import Callable
from datetime import UTC, datetime
from enum import StrEnum
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.ml.corpus import CorpusManifest, CorpusRecord

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from numpy.typing import NDArray

_EXPECTED_MATRIX_DIMENSIONS = 2
_MINIMUM_RECORDS = 3


class ReductionMode(StrEnum):
    CLUSTERING = "clustering"
    VISUALIZATION = "visualization"


class _ReductionModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class UMAPConfig(_ReductionModel):
    schema_version: int = 1
    n_neighbors: int = Field(default=15, ge=2)
    clustering_components: int = Field(default=5, ge=2, le=100)
    visualization_components: int = Field(default=2, ge=2, le=3)
    clustering_min_dist: float = Field(default=0.0, ge=0, le=1)
    visualization_min_dist: float = Field(default=0.1, ge=0, le=1)
    metric: str = Field(default="cosine", min_length=1)
    random_seed: int = 42
    low_memory: bool = True
    threads: int = Field(default=1, ge=1)
    training_sample_size: int | None = Field(default=100_000, ge=_MINIMUM_RECORDS)
    transform_batch_size: int = Field(default=8192, ge=1)
    trustworthiness_sample_size: int = Field(default=5000, ge=0)

    @model_validator(mode="after")
    def deterministic_threads(self) -> UMAPConfig:
        if self.threads != 1:
            msg = "UMAP threads must be 1 when random_seed is fixed for reproducibility"
            raise ValueError(msg)
        return self


class ReductionQuality(_ReductionModel):
    coordinate_variances: list[float]
    duplicate_coordinate_share: float = Field(ge=0, le=1)
    trustworthiness: float | None = Field(default=None, ge=0, le=1)


class ReductionArtifactManifest(_ReductionModel):
    schema_version: int = 1
    mode: ReductionMode
    corpus_manifest_path: str
    corpus_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_embeddings_path: str
    input_embeddings_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_records: int = Field(ge=0)
    output_records: int = Field(ge=0)
    input_dimensions: int = Field(ge=1)
    output_dimensions: int = Field(ge=2)
    reduced_path: str
    reduced_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_path: str
    model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_indices_path: str
    training_indices_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_records: int = Field(ge=1)
    config: UMAPConfig
    effective_n_neighbors: int = Field(ge=2)
    library: str
    library_version: str
    dtype: str = "float32"
    quality: ReductionQuality
    created_at: datetime


class DimensionalityReducer(Protocol):
    """Small injectable boundary around UMAP for deterministic unit tests."""

    library: str
    library_version: str

    def fit(self, vectors: NDArray[np.float32]) -> None: ...

    def transform(self, vectors: NDArray[np.float32]) -> NDArray[np.floating]: ...

    def dump(self, path: Path) -> None: ...


ReducerFactory = Callable[[ReductionMode, int, int, float, UMAPConfig], DimensionalityReducer]


class UMAPReducer:
    """Production adapter; model files are written but never deserialized here."""

    library = "umap-learn"

    def __init__(
        self,
        _mode: ReductionMode,
        components: int,
        neighbors: int,
        min_dist: float,
        config: UMAPConfig,
    ) -> None:
        try:
            import umap
        except ImportError as exc:  # pragma: no cover - optional ML dependency
            msg = "umap-learn is required; install the 'analysis' extra"
            raise RuntimeError(msg) from exc
        self.library_version = version("umap-learn")
        self._model = umap.UMAP(
            n_neighbors=neighbors,
            n_components=components,
            min_dist=min_dist,
            metric=config.metric,
            random_state=config.random_seed,
            low_memory=config.low_memory,
            n_jobs=config.threads,
            transform_seed=config.random_seed,
        )

    def fit(self, vectors: NDArray[np.float32]) -> None:
        self._model.fit(vectors)

    def transform(self, vectors: NDArray[np.float32]) -> NDArray[np.floating]:
        import numpy as np

        return np.asarray(self._model.transform(vectors), dtype=np.float32)

    def dump(self, path: Path) -> None:
        with path.open("wb") as target:
            pickle.dump(self._model, target, protocol=pickle.HIGHEST_PROTOCOL)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_corpus_contract(embeddings_path: Path, manifest_path: Path) -> CorpusManifest:
    manifest = CorpusManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    if _sha256_file(embeddings_path) != manifest.final_embeddings_sha256:
        msg = "final embeddings checksum does not match corpus manifest"
        raise ValueError(msg)
    corpus_path = Path(manifest.corpus_path)
    ids_path = Path(manifest.final_record_ids_path)
    if not corpus_path.is_file() or _sha256_file(corpus_path) != manifest.corpus_sha256:
        msg = "final corpus is missing or does not match corpus manifest"
        raise ValueError(msg)
    if not ids_path.is_file() or _sha256_file(ids_path) != manifest.final_record_ids_sha256:
        msg = "final record IDs are missing or do not match corpus manifest"
        raise ValueError(msg)
    count = 0
    with corpus_path.open(encoding="utf-8") as corpus, ids_path.open(encoding="utf-8") as ids:
        corpus_lines = (line for line in corpus if line.strip())
        id_lines = (line for line in ids if line.strip())
        for count, (record_line, id_line) in enumerate(zip(corpus_lines, id_lines, strict=True), start=1):
            record = CorpusRecord.model_validate_json(record_line)
            try:
                record_id = json.loads(id_line)
            except json.JSONDecodeError as exc:
                msg = f"invalid final record ID at line {count}"
                raise ValueError(msg) from exc
            if record.record_id != record_id:
                msg = f"final corpus and record ID alignment mismatch at index {count - 1}"
                raise ValueError(msg)
    if count != manifest.stats.output_records:
        msg = f"final corpus contains {count} records; manifest declares {manifest.stats.output_records}"
        raise ValueError(msg)
    return manifest


def deterministic_training_indices(n_records: int, sample_size: int | None, seed: int) -> list[int]:
    """Return stable sorted training indexes without changing output row order."""
    import numpy as np

    if n_records < _MINIMUM_RECORDS:
        msg = f"UMAP requires at least {_MINIMUM_RECORDS} records, got {n_records}"
        raise ValueError(msg)
    actual_size = n_records if sample_size is None else min(n_records, sample_size)
    if actual_size == n_records:
        return list(range(n_records))
    generator = np.random.default_rng(seed)
    return sorted(int(index) for index in generator.choice(n_records, size=actual_size, replace=False))


def _validate_input_matrix(vectors: NDArray[np.floating], expected_shape: tuple[int, int]) -> None:
    import numpy as np

    if vectors.shape != expected_shape or vectors.ndim != _EXPECTED_MATRIX_DIMENSIONS:
        msg = f"embedding matrix shape {vectors.shape} does not match expected {expected_shape}"
        raise ValueError(msg)
    for start in range(0, vectors.shape[0], 8192):
        chunk = vectors[start : start + 8192]
        if not np.isfinite(chunk).all():
            msg = "embedding matrix contains NaN or infinite values"
            raise ValueError(msg)
        if np.any(np.linalg.norm(chunk, axis=1) == 0):
            msg = "embedding matrix contains zero vectors"
            raise ValueError(msg)


def _validate_reduced(vectors: NDArray[np.floating], rows: int, dimensions: int) -> NDArray[np.float32]:
    import numpy as np

    result = np.asarray(vectors, dtype=np.float32)
    if result.shape != (rows, dimensions):
        msg = f"reducer returned shape {result.shape}; expected {(rows, dimensions)}"
        raise ValueError(msg)
    if not np.isfinite(result).all():
        msg = "reducer returned NaN or infinite coordinates"
        raise ValueError(msg)
    return result


def _quality(
    original_sample: NDArray[np.float32],
    reduced: NDArray[np.float32],
    config: UMAPConfig,
) -> ReductionQuality:
    import numpy as np

    variances = np.var(reduced, axis=0)
    if np.any(variances <= 0):
        msg = "reduced coordinates have zero variance"
        raise ValueError(msg)
    unique_rows = np.unique(reduced, axis=0).shape[0]
    duplicate_share = 1.0 - unique_rows / reduced.shape[0]
    trustworthiness_score: float | None = None
    quality_size = min(config.trustworthiness_sample_size, reduced.shape[0])
    if quality_size >= _MINIMUM_RECORDS:
        from sklearn.manifold import trustworthiness

        neighbors = min(5, (quality_size - 1) // 2)
        trustworthiness_score = float(
            trustworthiness(
                original_sample,
                reduced[:quality_size],
                n_neighbors=neighbors,
                metric=config.metric,
            ),
        )
    return ReductionQuality(
        coordinate_variances=[float(value) for value in variances],
        duplicate_coordinate_share=duplicate_share,
        trustworthiness=trustworthiness_score,
    )


def _mode_parameters(mode: ReductionMode, config: UMAPConfig) -> tuple[int, float]:
    if mode == ReductionMode.CLUSTERING:
        return config.clustering_components, config.clustering_min_dist
    return config.visualization_components, config.visualization_min_dist


def _run_mode(
    mode: ReductionMode,
    source: NDArray[np.float32],
    training_indices: list[int],
    output_dir: Path,
    corpus_manifest_path: Path,
    corpus_manifest: CorpusManifest,
    config: UMAPConfig,
    factory: ReducerFactory,
    *,
    output_records: int,
) -> ReductionArtifactManifest:
    import numpy as np
    from numpy.lib.format import open_memmap

    components, min_dist = _mode_parameters(mode, config)
    effective_neighbors = min(config.n_neighbors, len(training_indices) - 1)
    reducer = factory(mode, components, effective_neighbors, min_dist, config)
    training = np.asarray(source[training_indices], dtype=np.float32)
    reducer.fit(training)
    reduced_name = "clustering-reduced.npy" if mode == ReductionMode.CLUSTERING else "visualization-2d.npy"
    reduced_path = output_dir / reduced_name
    model_path = output_dir / f"{mode.value}-model.pkl"
    indices_path = output_dir / f"{mode.value}-training-indices.json"
    manifest_path = output_dir / f"{mode.value}-manifest.json"
    report_path = output_dir / f"{mode.value}-report.md"
    artifact_paths = (reduced_path, model_path, indices_path, manifest_path, report_path)
    temporary = {path: path.with_name(f".{path.name}.tmp") for path in artifact_paths}
    for path in temporary.values():
        path.unlink(missing_ok=True)
    try:
        target = open_memmap(
            temporary[reduced_path],
            mode="w+",
            dtype=np.float32,
            shape=(output_records, components),
        )
        for start in range(0, output_records, config.transform_batch_size):
            stop = min(output_records, start + config.transform_batch_size)
            transformed = reducer.transform(np.asarray(source[start:stop], dtype=np.float32))
            target[start:stop] = _validate_reduced(transformed, stop - start, components)
        target.flush()
        reduced = np.asarray(target)
        quality_size = min(config.trustworthiness_sample_size, output_records)
        original_sample = np.asarray(source[:quality_size], dtype=np.float32)
        quality = _quality(original_sample, reduced, config)
        del target
        reducer.dump(temporary[model_path])
        temporary[indices_path].write_text(f"{json.dumps(training_indices)}\n", encoding="utf-8")
        manifest = ReductionArtifactManifest(
            mode=mode,
            corpus_manifest_path=str(corpus_manifest_path),
            corpus_manifest_sha256=_sha256_file(corpus_manifest_path),
            input_embeddings_path=corpus_manifest.final_embeddings_path,
            input_embeddings_sha256=corpus_manifest.final_embeddings_sha256,
            input_records=corpus_manifest.stats.output_records,
            output_records=output_records,
            input_dimensions=corpus_manifest.dimensions,
            output_dimensions=components,
            reduced_path=str(reduced_path),
            reduced_sha256=_sha256_file(temporary[reduced_path]),
            model_path=str(model_path),
            model_sha256=_sha256_file(temporary[model_path]),
            training_indices_path=str(indices_path),
            training_indices_sha256=_sha256_file(temporary[indices_path]),
            training_records=len(training_indices),
            config=config,
            effective_n_neighbors=effective_neighbors,
            library=reducer.library,
            library_version=reducer.library_version,
            quality=quality,
            created_at=datetime.now(UTC),
        )
        temporary[manifest_path].write_text(f"{manifest.model_dump_json(indent=2)}\n", encoding="utf-8")
        temporary[report_path].write_text(_reduction_report(manifest), encoding="utf-8")
        for final in (reduced_path, model_path, indices_path, report_path, manifest_path):
            temporary[final].replace(final)
    except Exception:
        for path in temporary.values():
            path.unlink(missing_ok=True)
        raise
    return manifest


def reduce_dimensions(
    embeddings_path: Path,
    corpus_manifest_path: Path,
    output_dir: Path,
    *,
    config: UMAPConfig | None = None,
    modes: Sequence[ReductionMode] = (ReductionMode.CLUSTERING, ReductionMode.VISUALIZATION),
    reducer_factory: ReducerFactory | None = None,
    force: bool = False,
    limit: int | None = None,
) -> dict[ReductionMode, ReductionArtifactManifest]:
    """Create separately configured clustering and visualization projections."""
    import numpy as np

    active_config = config or UMAPConfig()
    selected_modes = tuple(dict.fromkeys(modes))
    if not selected_modes:
        msg = "at least one reduction mode is required"
        raise ValueError(msg)
    corpus_manifest = _validate_corpus_contract(embeddings_path, corpus_manifest_path)
    source = np.load(embeddings_path, mmap_mode="r", allow_pickle=False)
    _validate_input_matrix(source, (corpus_manifest.stats.output_records, corpus_manifest.dimensions))
    output_records = (
        corpus_manifest.stats.output_records
        if limit is None
        else min(limit, corpus_manifest.stats.output_records)
    )
    if limit is not None and limit < _MINIMUM_RECORDS:
        msg = f"limit must be at least {_MINIMUM_RECORDS}"
        raise ValueError(msg)
    training_indices = deterministic_training_indices(
        output_records,
        active_config.training_sample_size,
        active_config.random_seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    final_paths: list[Path] = []
    for mode in selected_modes:
        reduced_name = "clustering-reduced.npy" if mode == ReductionMode.CLUSTERING else "visualization-2d.npy"
        final_paths.extend(
            (
                output_dir / reduced_name,
                output_dir / f"{mode.value}-model.pkl",
                output_dir / f"{mode.value}-training-indices.json",
                output_dir / f"{mode.value}-manifest.json",
                output_dir / f"{mode.value}-report.md",
            ),
        )
    existing = next((path for path in final_paths if path.exists()), None)
    if existing is not None and not force:
        msg = f"refusing to overwrite existing dimensionality-reduction artifact: {existing}"
        raise FileExistsError(msg)
    factory = reducer_factory or UMAPReducer
    results = {}
    for mode in selected_modes:
        results[mode] = _run_mode(
            mode,
            source,
            training_indices,
            output_dir,
            corpus_manifest_path,
            corpus_manifest,
            active_config,
            factory,
            output_records=output_records,
        )
    return results


def _reduction_report(manifest: ReductionArtifactManifest) -> str:
    trustworthiness = (
        "not calculated"
        if manifest.quality.trustworthiness is None
        else f"{manifest.quality.trustworthiness:.4f}"
    )
    return f"""# {manifest.mode.value.title()} dimensionality reduction

- Input records: {manifest.input_records}
- Output records: {manifest.output_records}
- Input dimensions: {manifest.input_dimensions}
- Output dimensions: {manifest.output_dimensions}
- Training records: {manifest.training_records}
- Effective neighbours: {manifest.effective_n_neighbors}
- Library: `{manifest.library}=={manifest.library_version}`
- Reduced SHA-256: `{manifest.reduced_sha256}`
- Model SHA-256: `{manifest.model_sha256}`
- Duplicate coordinate share: {manifest.quality.duplicate_coordinate_share:.4%}
- Trustworthiness: {trustworthiness}

The model pickle is a local trusted artifact. Never deserialize it before verifying its checksum and origin. This
report contains no corpus text or provenance values.
"""
