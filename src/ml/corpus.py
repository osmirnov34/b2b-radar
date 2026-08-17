"""Build the checksum-bound corpus consumed by dimensionality reduction."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from src.ml.cleaning_dataset import DatasetCleaningManifest
from src.ml.models import DuplicateGroup
from src.ml.schemas import CleanedTextUnit, EmbeddingArtifactManifest, TextKind
from src.ml.semantic_deduplication import SemanticDeduplicationManifest

if TYPE_CHECKING:
    from collections.abc import Iterator


class _CorpusModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CorpusRecord(CleanedTextUnit):
    """Retained cleaned unit with stable corpus and source alignment metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    corpus_id: str = Field(pattern=r"^corpus:[0-9a-f]{64}$")
    cleaned_record_index: int = Field(ge=0)
    semantic_duplicate_count: int = Field(default=0, ge=0)


class CorpusStats(_CorpusModel):
    input_records: int = Field(ge=0)
    output_records: int = Field(ge=0)
    removed_semantic_duplicates: int = Field(ge=0)
    output_comments: int = Field(ge=0)
    output_replies: int = Field(ge=0)
    languages: dict[str, int]
    unique_videos: int = Field(ge=0)


class CorpusManifest(_CorpusModel):
    schema_version: int = 1
    records_path: str
    records_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cleaning_manifest_path: str
    cleaning_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    embeddings_path: str
    embeddings_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    embedding_manifest_path: str
    embedding_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    keep_indices_path: str
    keep_indices_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    groups_path: str
    groups_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    deduplication_manifest_path: str
    deduplication_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    corpus_path: str
    corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    final_embeddings_path: str
    final_embeddings_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    final_record_ids_path: str
    final_record_ids_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dimensions: int = Field(ge=1)
    dtype: str
    stats: CorpusStats
    created_at: datetime


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_cleaned_records(path: Path) -> Iterator[CleanedTextUnit]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                yield CleanedTextUnit.model_validate_json(line)
            except ValueError as exc:
                msg = f"invalid cleaned record at line {line_number}: {exc}"
                raise ValueError(msg) from exc


def _load_keep_indices(path: Path, n_records: int) -> list[int]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"invalid keep-indices JSON: {exc}"
        raise ValueError(msg) from exc
    if not isinstance(raw, list) or any(type(index) is not int for index in raw):
        msg = "keep indices must be a JSON array of integers"
        raise ValueError(msg)
    indices: list[int] = raw
    if indices != sorted(set(indices)):
        msg = "keep indices must be unique and strictly increasing"
        raise ValueError(msg)
    if any(index < 0 or index >= n_records for index in indices):
        msg = f"keep index is outside the valid range [0, {n_records})"
        raise ValueError(msg)
    return indices


def _load_groups(path: Path) -> list[DuplicateGroup]:
    groups: list[DuplicateGroup] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                groups.append(DuplicateGroup.model_validate_json(line))
            except ValueError as exc:
                msg = f"invalid semantic group at line {line_number}: {exc}"
                raise ValueError(msg) from exc
    return groups


def _validate_groups(groups: list[DuplicateGroup], keep_indices: list[int], n_records: int) -> dict[int, int]:
    kept = set(keep_indices)
    removed_to_representative: dict[int, int] = {}
    duplicate_counts: dict[int, int] = {}
    representatives: set[int] = set()
    for group in groups:
        representative = group.representative_index
        if representative < 0 or representative >= n_records or representative not in kept:
            msg = f"semantic group representative {representative} is not a valid kept index"
            raise ValueError(msg)
        if representative in representatives:
            msg = f"semantic representative occurs in multiple groups: {representative}"
            raise ValueError(msg)
        representatives.add(representative)
        if not group.duplicate_indices:
            msg = f"semantic group {representative} contains no duplicates"
            raise ValueError(msg)
        for duplicate in group.duplicate_indices:
            if duplicate < 0 or duplicate >= n_records:
                msg = f"semantic duplicate index {duplicate} is outside the corpus"
                raise ValueError(msg)
            if duplicate in kept or duplicate in removed_to_representative or duplicate == representative:
                msg = f"semantic duplicate index is kept or belongs to multiple groups: {duplicate}"
                raise ValueError(msg)
            removed_to_representative[duplicate] = representative
        duplicate_counts[representative] = len(group.duplicate_indices)
    expected_removed = set(range(n_records)) - kept
    if set(removed_to_representative) != expected_removed:
        msg = "semantic groups do not describe exactly the records excluded by keep indices"
        raise ValueError(msg)
    return duplicate_counts


def _validate_input_contracts(
    records_path: Path,
    cleaning_manifest_path: Path,
    embeddings_path: Path,
    embedding_manifest_path: Path,
    keep_indices_path: Path,
    groups_path: Path,
    deduplication_manifest_path: Path,
) -> tuple[EmbeddingArtifactManifest, SemanticDeduplicationManifest]:
    records_sha256 = _sha256_file(records_path)
    embeddings_sha256 = _sha256_file(embeddings_path)
    keep_sha256 = _sha256_file(keep_indices_path)
    groups_sha256 = _sha256_file(groups_path)
    cleaning = DatasetCleaningManifest.model_validate_json(cleaning_manifest_path.read_text(encoding="utf-8"))
    embedding = EmbeddingArtifactManifest.model_validate_json(embedding_manifest_path.read_text(encoding="utf-8"))
    deduplication = SemanticDeduplicationManifest.model_validate_json(
        deduplication_manifest_path.read_text(encoding="utf-8"),
    )
    if cleaning.output_sha256 != records_sha256:
        msg = "cleaned records checksum does not match cleaning manifest"
        raise ValueError(msg)
    if embedding.records_sha256 != records_sha256 or embedding.embeddings_sha256 != embeddings_sha256:
        msg = "records or embeddings checksum does not match embedding manifest"
        raise ValueError(msg)
    if (
        deduplication.records_sha256 != records_sha256
        or deduplication.embeddings_sha256 != embeddings_sha256
        or deduplication.keep_indices_sha256 != keep_sha256
        or deduplication.groups_sha256 != groups_sha256
    ):
        msg = "semantic-deduplication inputs do not match its manifest"
        raise ValueError(msg)
    return embedding, deduplication


def _validate_record_ids(records_path: Path, embedding_manifest: EmbeddingArtifactManifest) -> None:
    if embedding_manifest.record_ids_path is None or embedding_manifest.record_ids_sha256 is None:
        msg = "embedding manifest must reference the aligned record-ids artifact"
        raise ValueError(msg)
    record_ids_path = Path(embedding_manifest.record_ids_path)
    if not record_ids_path.is_file() or _sha256_file(record_ids_path) != embedding_manifest.record_ids_sha256:
        msg = "record IDs artifact is missing or does not match embedding manifest"
        raise ValueError(msg)
    with record_ids_path.open(encoding="utf-8") as ids:
        id_iterator = (line for line in ids if line.strip())
        count = 0
        aligned_records = zip(_iter_cleaned_records(records_path), id_iterator, strict=True)
        for count, (record, raw_id) in enumerate(aligned_records, start=1):
            try:
                persisted_id = json.loads(raw_id)
            except json.JSONDecodeError as exc:
                msg = f"invalid record ID JSON at line {count}"
                raise ValueError(msg) from exc
            if persisted_id != record.record_id:
                msg = f"record ID alignment mismatch at index {count - 1}"
                raise ValueError(msg)
    if count != embedding_manifest.n_records:
        msg = f"record IDs count {count} does not match embedding manifest {embedding_manifest.n_records}"
        raise ValueError(msg)


def _corpus_id(record_id: str) -> str:
    return f"corpus:{hashlib.sha256(record_id.encode()).hexdigest()}"


def _write_corpus(
    records_path: Path,
    corpus_path: Path,
    ids_path: Path,
    keep_indices: list[int],
    duplicate_counts: dict[int, int],
) -> tuple[Counter[TextKind], Counter[str], int]:
    keep_iterator = iter(keep_indices)
    next_keep = next(keep_iterator, None)
    kinds: Counter[TextKind] = Counter()
    languages: Counter[str] = Counter()
    videos: set[str] = set()
    written = 0
    with corpus_path.open("w", encoding="utf-8") as corpus, ids_path.open("w", encoding="utf-8") as ids:
        for index, record in enumerate(_iter_cleaned_records(records_path)):
            if index != next_keep:
                continue
            enriched = CorpusRecord(
                **record.model_dump(),
                corpus_id=_corpus_id(record.record_id),
                cleaned_record_index=index,
                semantic_duplicate_count=duplicate_counts.get(index, 0),
            )
            corpus.write(f"{enriched.model_dump_json()}\n")
            ids.write(f"{json.dumps(record.record_id, ensure_ascii=False)}\n")
            kinds[record.text_kind] += 1
            languages[record.detected_language] += 1
            if record.video_id:
                videos.add(record.video_id)
            written += 1
            next_keep = next(keep_iterator, None)
    if written != len(keep_indices):
        msg = f"wrote {written} corpus records for {len(keep_indices)} keep indices"
        raise ValueError(msg)
    return kinds, languages, len(videos)


def _write_embeddings(
    source_path: Path,
    target_path: Path,
    keep_indices: list[int],
    n_records: int,
    dimensions: int,
) -> str:
    import numpy as np
    from numpy.lib.format import open_memmap

    source = np.load(source_path, mmap_mode="r", allow_pickle=False)
    if source.shape != (n_records, dimensions):
        msg = "embedding matrix shape does not match embedding manifest"
        raise ValueError(msg)
    if keep_indices:
        target = open_memmap(target_path, mode="w+", dtype=np.float32, shape=(len(keep_indices), dimensions))
        chunk_size = 4096
        for start in range(0, len(keep_indices), chunk_size):
            chunk = keep_indices[start : start + chunk_size]
            target[start : start + len(chunk)] = source[chunk]
        target.flush()
        del target
    else:
        with target_path.open("wb") as empty_target:
            np.save(empty_target, np.empty((0, dimensions), dtype=np.float32))
    return str(source.dtype)


def _validate_completed_inputs(
    records_path: Path,
    embeddings_path: Path,
    embedding_manifest: EmbeddingArtifactManifest,
    source_dtype: str,
) -> None:
    if source_dtype != embedding_manifest.dtype or source_dtype != "float32":
        msg = f"embedding dtype {source_dtype!r} is not the manifest float32 contract"
        raise ValueError(msg)
    if _sha256_file(records_path) != embedding_manifest.records_sha256:
        msg = "cleaned records changed while building final corpus"
        raise ValueError(msg)
    if _sha256_file(embeddings_path) != embedding_manifest.embeddings_sha256:
        msg = "embeddings changed while building final corpus"
        raise ValueError(msg)


def build_final_corpus(
    records_path: Path,
    embeddings_path: Path,
    keep_indices_path: Path,
    *,
    cleaning_manifest_path: Path,
    embedding_manifest_path: Path,
    groups_path: Path,
    deduplication_manifest_path: Path,
    output_dir: Path,
    force: bool = False,
) -> CorpusManifest:
    """Apply verified semantic keep indexes and publish aligned corpus artifacts."""
    embedding_manifest, deduplication_manifest = _validate_input_contracts(
        records_path,
        cleaning_manifest_path,
        embeddings_path,
        embedding_manifest_path,
        keep_indices_path,
        groups_path,
        deduplication_manifest_path,
    )
    _validate_record_ids(records_path, embedding_manifest)
    if deduplication_manifest.result.n_input != embedding_manifest.n_records:
        msg = "deduplication and embedding manifests disagree on input record count"
        raise ValueError(msg)
    keep_indices = _load_keep_indices(keep_indices_path, embedding_manifest.n_records)
    groups = _load_groups(groups_path)
    duplicate_counts = _validate_groups(groups, keep_indices, embedding_manifest.n_records)
    if (
        len(keep_indices) != deduplication_manifest.result.n_kept
        or embedding_manifest.n_records - len(keep_indices) != deduplication_manifest.result.n_removed
    ):
        msg = "keep indices counts do not match semantic-deduplication manifest"
        raise ValueError(msg)

    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = output_dir / "final-corpus.jsonl"
    final_embeddings_path = output_dir / "final-embeddings.npy"
    final_ids_path = output_dir / "final-record-ids.jsonl"
    manifest_path = output_dir / "corpus-manifest.json"
    report_path = output_dir / "corpus-report.md"
    finals = (corpus_path, final_embeddings_path, final_ids_path, manifest_path, report_path)
    existing = next((path for path in finals if path.exists()), None)
    if existing is not None and not force:
        msg = f"refusing to overwrite existing corpus artifact: {existing}"
        raise FileExistsError(msg)
    temporary = {path: path.with_name(f".{path.name}.tmp") for path in finals}
    for path in temporary.values():
        path.unlink(missing_ok=True)
    try:
        kinds, languages, unique_videos = _write_corpus(
            records_path,
            temporary[corpus_path],
            temporary[final_ids_path],
            keep_indices,
            duplicate_counts,
        )
        source_dtype = _write_embeddings(
            embeddings_path,
            temporary[final_embeddings_path],
            keep_indices,
            embedding_manifest.n_records,
            embedding_manifest.dimensions,
        )
        _validate_completed_inputs(records_path, embeddings_path, embedding_manifest, source_dtype)
        stats = CorpusStats(
            input_records=embedding_manifest.n_records,
            output_records=len(keep_indices),
            removed_semantic_duplicates=embedding_manifest.n_records - len(keep_indices),
            output_comments=kinds[TextKind.COMMENT],
            output_replies=kinds[TextKind.REPLY],
            languages=dict(languages),
            unique_videos=unique_videos,
        )
        manifest = CorpusManifest(
            records_path=str(records_path),
            records_sha256=_sha256_file(records_path),
            cleaning_manifest_path=str(cleaning_manifest_path),
            cleaning_manifest_sha256=_sha256_file(cleaning_manifest_path),
            embeddings_path=str(embeddings_path),
            embeddings_sha256=_sha256_file(embeddings_path),
            embedding_manifest_path=str(embedding_manifest_path),
            embedding_manifest_sha256=_sha256_file(embedding_manifest_path),
            keep_indices_path=str(keep_indices_path),
            keep_indices_sha256=_sha256_file(keep_indices_path),
            groups_path=str(groups_path),
            groups_sha256=_sha256_file(groups_path),
            deduplication_manifest_path=str(deduplication_manifest_path),
            deduplication_manifest_sha256=_sha256_file(deduplication_manifest_path),
            corpus_path=str(corpus_path),
            corpus_sha256=_sha256_file(temporary[corpus_path]),
            final_embeddings_path=str(final_embeddings_path),
            final_embeddings_sha256=_sha256_file(temporary[final_embeddings_path]),
            final_record_ids_path=str(final_ids_path),
            final_record_ids_sha256=_sha256_file(temporary[final_ids_path]),
            dimensions=embedding_manifest.dimensions,
            dtype="float32",
            stats=stats,
            created_at=datetime.now(UTC),
        )
        temporary[manifest_path].write_text(f"{manifest.model_dump_json(indent=2)}\n", encoding="utf-8")
        temporary[report_path].write_text(_corpus_report(manifest), encoding="utf-8")
        for final in (corpus_path, final_embeddings_path, final_ids_path, report_path, manifest_path):
            temporary[final].replace(final)
    except Exception:
        for path in temporary.values():
            path.unlink(missing_ok=True)
        raise
    return manifest


def _corpus_report(manifest: CorpusManifest) -> str:
    stats = manifest.stats
    languages = "\n".join(f"- `{language}`: {count}" for language, count in sorted(stats.languages.items()))
    return f"""# Final corpus

- Input records: {stats.input_records}
- Output records: {stats.output_records}
- Removed semantic duplicates: {stats.removed_semantic_duplicates}
- Output comments/replies: {stats.output_comments}/{stats.output_replies}
- Unique videos: {stats.unique_videos}
- Embedding shape: {stats.output_records} x {manifest.dimensions}
- Embedding dtype: `{manifest.dtype}`
- Corpus SHA-256: `{manifest.corpus_sha256}`
- Embeddings SHA-256: `{manifest.final_embeddings_sha256}`

## Languages

{languages or '- None.'}

This report contains aggregate counts and checksums only. It contains no comment, reply, author, query, URL, channel,
or video values.
"""
