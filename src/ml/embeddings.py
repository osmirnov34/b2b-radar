"""Streaming multilingual embedding generation with resumable artifacts."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, ConfigDict, Field

from src.ml.config import EmbeddingConfig
from src.ml.schemas import CleanedTextUnit, EmbeddingArtifactManifest

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

    import numpy as np
    from numpy.typing import NDArray


class TextEncoder(Protocol):
    """Minimal encoder interface used by production and lightweight tests."""

    @property
    def device(self) -> str: ...

    def encode(self, texts: Sequence[str]) -> NDArray[np.floating]: ...


_EXPECTED_VECTOR_DIMENSIONS = 2


class SentenceTransformerEncoder:
    """Lazy adapter around sentence-transformers."""

    def __init__(self, config: EmbeddingConfig) -> None:
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional heavyweight dependency
            msg = "sentence-transformers and torch are required; install the 'analysis' extra"
            raise RuntimeError(msg) from exc

        torch.set_num_threads(config.threads)
        torch.manual_seed(config.seed)
        requested_device = None if config.device == "auto" else config.device
        if config.device == "cuda" and not torch.cuda.is_available():
            msg = "CUDA was requested but is not available"
            raise RuntimeError(msg)
        self._model = SentenceTransformer(
            config.model_name,
            revision=config.model_revision,
            device=requested_device,
        )
        self._model.max_seq_length = min(self._model.max_seq_length, config.max_seq_length)
        self._config = config

    @property
    def device(self) -> str:
        return str(self._model.device)

    def encode(self, texts: Sequence[str]) -> NDArray[np.floating]:
        import numpy as np

        prefixed = [f"{embedding_prompt(self._config)}{text}" for text in texts]
        vectors = self._model.encode(
            prefixed,
            batch_size=len(prefixed),
            show_progress_bar=False,
            normalize_embeddings=self._config.normalize,
        )
        return np.asarray(vectors, dtype=np.float32)


class EmbeddingProgress(BaseModel):
    """Checkpoint for a partially written embedding matrix."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    records_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    n_records: int = Field(ge=1)
    dimensions: int = Field(ge=1)
    processed_records: int = Field(ge=0)
    device: str
    limit: int | None = Field(default=None, ge=1)


def embedding_prompt(config: EmbeddingConfig) -> str:
    """Resolve an explicit prompt or a conservative model-family default."""
    if config.prompt_prefix is not None:
        return config.prompt_prefix
    name = config.model_name.casefold()
    if "e5" in name:
        return "query: "
    if "frida" in name:
        return "categorize_topic: "
    return ""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_sha256(config: EmbeddingConfig) -> str:
    payload = config.model_dump_json(exclude_none=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _iter_records(path: Path, *, limit: int | None = None) -> Iterator[CleanedTextUnit]:
    emitted = 0
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            if limit is not None and emitted >= limit:
                return
            try:
                record = CleanedTextUnit.model_validate_json(line)
            except ValueError as exc:
                msg = f"invalid cleaned record at line {line_number}: {exc}"
                raise ValueError(msg) from exc
            emitted += 1
            yield record


def _inspect_records(records_path: Path, record_ids_path: Path, *, limit: int | None) -> int:
    seen: set[str] = set()
    count = 0
    with record_ids_path.open("w", encoding="utf-8") as target:
        for record in _iter_records(records_path, limit=limit):
            if record.record_id in seen:
                msg = f"duplicate record_id in cleaned dataset: {record.record_id!r}"
                raise ValueError(msg)
            seen.add(record.record_id)
            target.write(f"{json.dumps(record.record_id, ensure_ascii=False)}\n")
            count += 1
    if count == 0:
        msg = "cleaned dataset contains no records"
        raise ValueError(msg)
    return count


def _validate_vectors(vectors: NDArray[np.floating], expected_rows: int, *, normalized: bool) -> NDArray[np.float32]:
    import numpy as np

    result = np.asarray(vectors, dtype=np.float32)
    if result.ndim != _EXPECTED_VECTOR_DIMENSIONS or result.shape[0] != expected_rows or result.shape[1] == 0:
        msg = f"encoder returned shape={result.shape}; expected ({expected_rows}, dimensions)"
        raise ValueError(msg)
    if not np.isfinite(result).all():
        msg = "encoder returned NaN or infinite values"
        raise ValueError(msg)
    norms = np.linalg.norm(result, axis=1)
    if np.any(norms == 0):
        msg = "encoder returned zero vectors"
        raise ValueError(msg)
    if normalized and not np.allclose(norms, 1.0, rtol=1e-3, atol=1e-4):
        msg = "encoder returned non-unit vectors although normalization is enabled"
        raise ValueError(msg)
    return result


def _batches(records_path: Path, batch_size: int, *, skip: int, limit: int | None) -> Iterator[list[str]]:
    batch: list[str] = []
    for index, record in enumerate(_iter_records(records_path, limit=limit)):
        if index < skip:
            continue
        batch.append(record.clean_text)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def generate_embeddings(
    records_path: Path,
    output_dir: Path,
    *,
    config: EmbeddingConfig | None = None,
    encoder: TextEncoder | None = None,
    resume: bool = False,
    force: bool = False,
    limit: int | None = None,
) -> EmbeddingArtifactManifest:
    """Generate aligned float32 embeddings without retaining the corpus in memory."""
    import numpy as np
    from numpy.lib.format import open_memmap

    if limit is not None and limit < 1:
        msg = "limit must be at least 1"
        raise ValueError(msg)
    active_config = config or EmbeddingConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = output_dir / "embeddings.npy"
    record_ids_path = output_dir / "record-ids.jsonl"
    manifest_path = output_dir / "embedding-manifest.json"
    report_path = output_dir / "embedding-report.md"
    partial_embeddings = output_dir / ".embeddings.partial.npy"
    partial_ids = output_dir / ".record-ids.partial.jsonl"
    progress_path = output_dir / ".embedding-progress.json"
    finals = (embeddings_path, record_ids_path, manifest_path, report_path)
    partials = (partial_embeddings, partial_ids, progress_path)

    existing_final = next((path for path in finals if path.exists()), None)
    if existing_final is not None and not force:
        msg = f"refusing to overwrite existing embedding artifact: {existing_final}"
        raise FileExistsError(msg)
    if force:
        for path in (*finals, *partials):
            path.unlink(missing_ok=True)
    elif not resume and any(path.exists() for path in partials):
        msg = "partial embedding artifacts exist; use resume=True or force=True"
        raise FileExistsError(msg)

    records_sha256 = _sha256_file(records_path)
    config_sha256 = _config_sha256(active_config)
    active_encoder = encoder or SentenceTransformerEncoder(active_config)

    if resume:
        if not all(path.exists() for path in partials):
            msg = "cannot resume: partial matrix, record IDs, and progress checkpoint are all required"
            raise FileNotFoundError(msg)
        progress = EmbeddingProgress.model_validate_json(progress_path.read_text(encoding="utf-8"))
        if progress.records_sha256 != records_sha256 or progress.config_sha256 != config_sha256:
            msg = "cannot resume with changed input data or embedding configuration"
            raise ValueError(msg)
        if progress.limit != limit:
            msg = "cannot resume with a different record limit"
            raise ValueError(msg)
        if progress.device != active_encoder.device:
            msg = f"cannot resume on device {active_encoder.device!r}; checkpoint uses {progress.device!r}"
            raise ValueError(msg)
        n_records = progress.n_records
        processed = progress.processed_records
        matrix = open_memmap(partial_embeddings, mode="r+")
        if matrix.shape != (n_records, progress.dimensions) or matrix.dtype != np.float32:
            msg = "partial embedding matrix does not match its checkpoint"
            raise ValueError(msg)
    else:
        n_records = _inspect_records(records_path, partial_ids, limit=limit)
        processed = 0
        matrix = None

    for texts in _batches(records_path, active_config.batch_size, skip=processed, limit=limit):
        vectors = _validate_vectors(active_encoder.encode(texts), len(texts), normalized=active_config.normalize)
        if matrix is None:
            matrix = open_memmap(
                partial_embeddings,
                mode="w+",
                dtype=np.float32,
                shape=(n_records, vectors.shape[1]),
            )
        elif vectors.shape[1] != matrix.shape[1]:
            msg = f"encoder dimension changed from {matrix.shape[1]} to {vectors.shape[1]}"
            raise ValueError(msg)
        matrix[processed : processed + len(texts)] = vectors
        matrix.flush()
        processed += len(texts)
        progress = EmbeddingProgress(
            records_sha256=records_sha256,
            config_sha256=config_sha256,
            n_records=n_records,
            dimensions=matrix.shape[1],
            processed_records=processed,
            device=active_encoder.device,
            limit=limit,
        )
        _atomic_write(progress_path, f"{progress.model_dump_json(indent=2)}\n")

    if matrix is None or processed != n_records:
        msg = f"embedding run stopped at {processed} of {n_records} records"
        raise RuntimeError(msg)
    dimensions = matrix.shape[1]
    del matrix
    if _sha256_file(records_path) != records_sha256:
        msg = "cleaned dataset changed while embeddings were being generated"
        raise ValueError(msg)
    partial_embeddings.replace(embeddings_path)
    partial_ids.replace(record_ids_path)
    embeddings_sha256 = _sha256_file(embeddings_path)
    record_ids_sha256 = _sha256_file(record_ids_path)
    manifest = EmbeddingArtifactManifest(
        records_sha256=records_sha256,
        embeddings_sha256=embeddings_sha256,
        n_records=n_records,
        dimensions=dimensions,
        model_name=active_config.model_name,
        model_revision=active_config.model_revision,
        normalized=active_config.normalize,
        record_ids_path=str(record_ids_path),
        record_ids_sha256=record_ids_sha256,
        device=active_encoder.device,
        batch_size=active_config.batch_size,
        max_seq_length=active_config.max_seq_length,
        prompt_prefix=embedding_prompt(active_config),
        config_sha256=config_sha256,
        created_at=datetime.now(UTC),
    )
    _atomic_write(manifest_path, f"{manifest.model_dump_json(indent=2)}\n")
    _atomic_write(
        report_path,
        "\n".join(
            (
                "# Embedding generation report",
                "",
                f"- Records: {n_records}",
                f"- Dimensions: {dimensions}",
                f"- Model: `{active_config.model_name}`",
                f"- Device: `{active_encoder.device}`",
                f"- Normalized: {active_config.normalize}",
                f"- Input SHA-256: `{records_sha256}`",
                "",
            ),
        ),
    )
    progress_path.unlink(missing_ok=True)
    return manifest
