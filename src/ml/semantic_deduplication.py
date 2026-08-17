"""ANN-based semantic near-duplicate grouping for normalized embeddings."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, ConfigDict, Field

from src.ml.config import DeduplicationConfig
from src.ml.models import DeduplicationResult, DeduplicationStats, DuplicateGroup, DuplicatePair
from src.ml.schemas import CleanedTextUnit, EmbeddingArtifactManifest, TextKind

_MIN_VECTORS_FOR_NEIGHBOURS = 2
_EXPECTED_VECTOR_DIMENSIONS = 2

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

    import numpy as np
    from numpy.typing import NDArray


class CandidateIndex(Protocol):
    """Candidate generator; only returned edges are threshold-checked by the caller."""

    def candidates(self, vectors: NDArray[np.float32], neighbors: int) -> Iterable[tuple[int, int, float]]: ...


class HnswCandidateIndex:
    """Approximate cosine-neighbour candidates backed by hnswlib."""

    def __init__(self, config: DeduplicationConfig) -> None:
        self.config = config

    def candidates(self, vectors: NDArray[np.float32], neighbors: int) -> Iterable[tuple[int, int, float]]:
        try:
            import hnswlib
        except ImportError as exc:  # pragma: no cover - depends on optional ML environment
            msg = "hnswlib is required for ANN deduplication; install the 'analysis' extra"
            raise RuntimeError(msg) from exc
        count, dimensions = vectors.shape
        if count < _MIN_VECTORS_FOR_NEIGHBOURS:
            return
        index = hnswlib.Index(space="cosine", dim=dimensions)
        index.init_index(
            max_elements=count,
            ef_construction=self.config.ann_ef_construction,
            M=self.config.ann_m,
            random_seed=self.config.random_seed,
        )
        index.add_items(vectors, num_threads=1)
        index.set_ef(max(self.config.ann_ef_search, neighbors))
        labels, distances = index.knn_query(vectors, k=min(neighbors + 1, count), num_threads=1)
        for source, (row_labels, row_distances) in enumerate(zip(labels, distances, strict=True)):
            for target, distance in zip(row_labels, row_distances, strict=True):
                target_index = int(target)
                if target_index != source:
                    left, right = sorted((source, target_index))
                    yield left, right, 1.0 - float(distance)


class ExhaustiveCandidateIndex:
    """Exact small-data backend for tests and calibration, never the 260k production corpus."""

    def candidates(self, vectors: NDArray[np.float32], _neighbors: int) -> Iterable[tuple[int, int, float]]:
        import numpy as np

        count = vectors.shape[0]
        for source in range(count):
            similarities = vectors[source + 1 :] @ vectors[source]
            if not similarities.size:
                continue
            for offset in np.argsort(similarities)[::-1]:
                yield source, source + 1 + int(offset), float(similarities[offset])


def _normalize_vectors(embeddings: NDArray[np.floating]) -> NDArray[np.float32]:
    import numpy as np

    vectors = np.asarray(embeddings, dtype=np.float32)
    if vectors.ndim != _EXPECTED_VECTOR_DIMENSIONS:
        msg = f"embeddings must be a 2D matrix, got shape={vectors.shape}"
        raise ValueError(msg)
    if not np.isfinite(vectors).all():
        msg = "embeddings contain non-finite values"
        raise ValueError(msg)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        msg = "embeddings contain zero vectors"
        raise ValueError(msg)
    return np.asarray(vectors / norms, dtype=np.float32)


def semantic_deduplicate(
    embeddings: NDArray[np.floating],
    config: DeduplicationConfig,
    *,
    text_kinds: Sequence[TextKind] | None = None,
    candidate_index: CandidateIndex | None = None,
) -> DeduplicationResult:
    """Group ANN candidates by cosine threshold with earliest-index representatives."""
    vectors = _normalize_vectors(embeddings)
    count = vectors.shape[0]
    if text_kinds is not None and len(text_kinds) != count:
        msg = "text_kinds length must match embeddings rows"
        raise ValueError(msg)
    if not config.enabled:
        return DeduplicationResult(
            keep_indices=list(range(count)),
            stats=DeduplicationStats(n_input=count, n_kept=count, n_removed=0, threshold=config.threshold),
        )
    backend = candidate_index
    if backend is None:
        if config.backend == "exhaustive" and count > config.exhaustive_max_records:
            msg = (
                f"exhaustive backend is limited to {config.exhaustive_max_records} records; "
                "use hnsw for the production corpus"
            )
            raise ValueError(msg)
        backend = HnswCandidateIndex(config) if config.backend == "hnsw" else ExhaustiveCandidateIndex()
    parent = list(range(count))

    def find(index: int) -> int:
        root = index
        while parent[root] != root:
            root = parent[root]
        while parent[index] != root:
            parent[index], index = root, parent[index]
        return root

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        earlier, later = sorted((left_root, right_root))
        parent[later] = earlier

    candidate_sets: list[tuple[list[int], NDArray[np.float32]]] = [(list(range(count)), vectors)]
    if config.separate_text_kinds and text_kinds is not None:
        candidate_sets = []
        for text_kind in TextKind:
            global_indices = [index for index, value in enumerate(text_kinds) if value == text_kind]
            if global_indices:
                candidate_sets.append((global_indices, vectors[global_indices]))
    for global_indices, candidate_vectors in candidate_sets:
        for local_left, local_right, similarity in backend.candidates(candidate_vectors, config.ann_neighbors):
            left, right = global_indices[local_left], global_indices[local_right]
            if not (0 <= left < right < count):
                msg = f"candidate index returned invalid edge ({left}, {right}) for {count} vectors"
                raise ValueError(msg)
            if similarity >= config.threshold:
                union(left, right)

    grouped: dict[int, list[int]] = defaultdict(list)
    for index in range(count):
        grouped[find(index)].append(index)
    groups = [
        DuplicateGroup(representative_index=representative, duplicate_indices=members[1:])
        for representative, members in sorted(grouped.items())
        if len(members) > 1
    ]
    removed = {index for group in groups for index in group.duplicate_indices}
    keep_indices = [index for index in range(count) if index not in removed]
    pairs = [
        DuplicatePair(
            representative_index=group.representative_index,
            duplicate_index=duplicate,
            similarity=float(vectors[group.representative_index] @ vectors[duplicate]),
        )
        for group in groups
        for duplicate in group.duplicate_indices
    ][: config.sample_pairs]
    return DeduplicationResult(
        keep_indices=keep_indices,
        sample_pairs=pairs,
        groups=groups,
        stats=DeduplicationStats(
            n_input=count,
            n_kept=len(keep_indices),
            n_removed=len(removed),
            threshold=config.threshold,
        ),
    )


class SemanticDeduplicationManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    records_path: str
    records_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    embeddings_path: str
    embeddings_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    embedding_manifest_path: str
    config: DeduplicationConfig
    keep_indices_path: str
    keep_indices_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    groups_path: str
    groups_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    result: DeduplicationStats
    created_at: datetime


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_semantic_deduplication(
    records_path: Path,
    embeddings_path: Path,
    embedding_manifest_path: Path,
    output_dir: Path,
    *,
    config: DeduplicationConfig | None = None,
    force: bool = False,
) -> SemanticDeduplicationManifest:
    """Run semantic deduplication for cleaned units and persist index-only artifacts."""
    import numpy as np

    active_config = config or DeduplicationConfig()
    keep_path = output_dir / "keep-indices.json"
    groups_path = output_dir / "semantic-groups.jsonl"
    manifest_path = output_dir / "semantic-deduplication-manifest.json"
    report_path = output_dir / "semantic-deduplication-report.md"
    existing = next((path for path in (keep_path, groups_path, manifest_path, report_path) if path.exists()), None)
    if existing is not None and not force:
        msg = f"refusing to overwrite existing semantic-deduplication artifact: {existing}"
        raise FileExistsError(msg)
    embedding_manifest = EmbeddingArtifactManifest.model_validate_json(
        embedding_manifest_path.read_text(encoding="utf-8"),
    )
    records_sha256 = _sha256_file(records_path)
    embeddings_sha256 = _sha256_file(embeddings_path)
    if records_sha256 != embedding_manifest.records_sha256:
        msg = "cleaned records checksum does not match embedding manifest"
        raise ValueError(msg)
    if embeddings_sha256 != embedding_manifest.embeddings_sha256:
        msg = "embeddings checksum does not match embedding manifest"
        raise ValueError(msg)
    text_kinds: list[TextKind] = []
    with records_path.open(encoding="utf-8") as source:
        text_kinds = [
            CleanedTextUnit.model_validate_json(line).text_kind
            for line in source
            if line.strip()
        ]
    embeddings = np.load(embeddings_path, allow_pickle=False)
    if embeddings.shape[0] != len(text_kinds):
        msg = f"embeddings rows {embeddings.shape[0]} do not match cleaned records {len(text_kinds)}"
        raise ValueError(msg)
    if embeddings.shape != (embedding_manifest.n_records, embedding_manifest.dimensions):
        msg = f"embeddings shape {embeddings.shape} does not match embedding manifest"
        raise ValueError(msg)
    result = semantic_deduplicate(embeddings, active_config, text_kinds=text_kinds)
    output_dir.mkdir(parents=True, exist_ok=True)
    keep_path.write_text(f"{json.dumps(result.keep_indices)}\n", encoding="utf-8")
    with groups_path.open("w", encoding="utf-8") as target:
        for group in result.groups:
            target.write(f"{group.model_dump_json()}\n")
    if _sha256_file(records_path) != records_sha256 or _sha256_file(embeddings_path) != embeddings_sha256:
        keep_path.unlink(missing_ok=True)
        groups_path.unlink(missing_ok=True)
        msg = "records or embeddings changed while semantic deduplication was running"
        raise ValueError(msg)
    manifest = SemanticDeduplicationManifest(
        records_path=str(records_path),
        records_sha256=records_sha256,
        embeddings_path=str(embeddings_path),
        embeddings_sha256=embeddings_sha256,
        embedding_manifest_path=str(embedding_manifest_path),
        config=active_config,
        keep_indices_path=str(keep_path),
        keep_indices_sha256=_sha256_file(keep_path),
        groups_path=str(groups_path),
        groups_sha256=_sha256_file(groups_path),
        result=result.stats,
        created_at=datetime.now(UTC),
    )
    manifest_path.write_text(f"{manifest.model_dump_json(indent=2)}\n", encoding="utf-8")
    report_path.write_text(
        "# Semantic deduplication\n\n"
        f"- Input: {result.stats.n_input}\n"
        f"- Kept: {result.stats.n_kept}\n"
        f"- Removed: {result.stats.n_removed}\n"
        f"- Threshold: {result.stats.threshold}\n"
        f"- Backend: `{active_config.backend}`\n"
        f"- ANN neighbours: {active_config.ann_neighbors}\n"
        f"- Separate comment/reply roles: {active_config.separate_text_kinds}\n\n"
        "Artifacts contain indexes and similarities only; manual pair review must be joined locally.\n",
        encoding="utf-8",
    )
    return manifest
