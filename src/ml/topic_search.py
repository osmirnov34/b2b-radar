"""Semantic search for user-defined themes over a checksum-bound ML corpus."""

from __future__ import annotations

import hashlib
import heapq
import re
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from src.ml.config import EmbeddingConfig
from src.ml.corpus import CorpusRecord
from src.ml.embeddings import SentenceTransformerEncoder, TextEncoder
from src.ml.reporting import AnalysisArtifacts, ProblemSignalConfig, match_problem_signals
from src.ml.schemas import EmbeddingArtifactManifest

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

_MAX_QUERY_LENGTH = 500
_MATRIX_DIMENSIONS = 2
_YOUTUBE_VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{6,20}$")


class _SearchModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TopicSearchConfig(_SearchModel):
    """Configure semantic retrieval and bounded human-readable evidence."""

    schema_version: int = 1
    minimum_similarity: float = Field(default=0.5, ge=-1, le=1)
    maximum_matches: int = Field(default=500, ge=1, le=10_000)
    evidence_per_topic: int = Field(default=5, ge=1, le=50)
    similarity_batch_size: int = Field(default=16_384, ge=128, le=1_000_000)
    include_outliers: bool = False
    problem_signals: ProblemSignalConfig = Field(default_factory=ProblemSignalConfig)


class TopicSearchEvidence(_SearchModel):
    """Expose one relevant source row without its author identity."""

    record_index: int = Field(ge=0)
    topic_id: int | None = Field(default=None, ge=0)
    topic_name: str | None = None
    semantic_similarity: float = Field(ge=-1, le=1)
    cluster_confidence: float = Field(ge=0, le=1)
    problem_signals: tuple[str, ...]
    text: str
    text_kind: str
    published_at: datetime | None = None
    video_title: str
    video_channel: str
    video_url: str


class TopicSearchGroup(_SearchModel):
    """Summarize the query-matched subcorpus assigned to one persisted topic."""

    topic_id: int | None = Field(default=None, ge=0)
    topic_name: str
    relevant_records: int = Field(ge=1)
    problem_records: int = Field(ge=0)
    problem_share: float = Field(ge=0, le=1)
    unique_videos: int = Field(ge=0)
    maximum_similarity: float = Field(ge=-1, le=1)
    mean_similarity: float = Field(ge=-1, le=1)
    evidence: tuple[TopicSearchEvidence, ...]


class TopicSearchResult(_SearchModel):
    """Contain one reproducible restricted search result."""

    result_schema_version: int = 1
    run_id: str
    pipeline_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    embeddings_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    query: str
    config: TopicSearchConfig
    model_name: str
    model_revision: str | None = None
    corpus_records: int = Field(ge=0)
    relevant_records: int = Field(ge=0)
    problem_records: int = Field(ge=0)
    groups: tuple[TopicSearchGroup, ...]
    classification: str = "restricted_user_topic_search"
    searched_at: datetime


class TopicSearchExportManifest(_SearchModel):
    """Bind persisted restricted search output to its source result."""

    export_schema_version: int = 1
    run_id: str
    pipeline_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    query_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    result_path: str
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    classification: str = "restricted_user_topic_search"
    created_at: datetime


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_path(run_dir: Path, raw_path: str, expected_sha256: str) -> Path:
    declared = Path(raw_path)
    path = declared.resolve()
    if not path.is_relative_to(run_dir.resolve()) or declared.is_symlink():
        msg = f"topic-search artifact escapes the selected run or is unsafe: {path.name}"
        raise ValueError(msg)
    if not path.is_file():
        msg = f"topic-search artifact is missing: {path.name}"
        raise FileNotFoundError(msg)
    if _sha256(path) != expected_sha256:
        msg = f"topic-search artifact checksum mismatch: {path.name}"
        raise ValueError(msg)
    return path


def _embedding_contract(artifacts: AnalysisArtifacts) -> tuple[Path, EmbeddingArtifactManifest]:
    matrix_path = _verified_path(
        artifacts.run_dir,
        artifacts.corpus.final_embeddings_path,
        artifacts.corpus.final_embeddings_sha256,
    )
    manifest_path = _verified_path(
        artifacts.run_dir,
        artifacts.corpus.embedding_manifest_path,
        artifacts.corpus.embedding_manifest_sha256,
    )
    manifest = EmbeddingArtifactManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    if manifest.dimensions != artifacts.corpus.dimensions:
        msg = "topic-search embedding dimensions disagree with the final corpus"
        raise ValueError(msg)
    if not manifest.normalized:
        msg = "topic search requires normalized source embeddings"
        raise ValueError(msg)
    return matrix_path, manifest


def _query_encoder_config(manifest: EmbeddingArtifactManifest) -> EmbeddingConfig:
    return EmbeddingConfig(
        model_name=manifest.model_name,
        model_revision=manifest.model_revision,
        batch_size=1,
        max_seq_length=manifest.max_seq_length or 512,
        normalize=True,
        prompt_prefix=manifest.prompt_prefix,
    )


def _query_vector(encoder: TextEncoder, query: str, dimensions: int) -> NDArray[np.float32]:
    import numpy as np

    vectors = np.asarray(encoder.encode([query]), dtype=np.float32)
    if vectors.shape != (1, dimensions) or not np.isfinite(vectors).all():
        msg = f"topic-search encoder returned invalid shape or values: {vectors.shape}"
        raise ValueError(msg)
    norm = float(np.linalg.norm(vectors[0]))
    if norm == 0:
        msg = "topic-search encoder returned a zero vector"
        raise ValueError(msg)
    return np.asarray(vectors[0] / norm, dtype=np.float32)


def _top_matches(
    matrix_path: Path,
    query_vector: NDArray[np.float32],
    labels: NDArray[np.int64],
    config: TopicSearchConfig,
) -> list[tuple[int, float]]:
    import numpy as np

    matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
    if (
        matrix.ndim != _MATRIX_DIMENSIONS
        or matrix.shape != (len(labels), len(query_vector))
        or not np.issubdtype(matrix.dtype, np.floating)
    ):
        msg = f"topic-search embeddings are not aligned: shape={matrix.shape}, labels={len(labels)}"
        raise ValueError(msg)
    heap: list[tuple[float, int]] = []
    for start in range(0, len(matrix), config.similarity_batch_size):
        batch = np.asarray(matrix[start : start + config.similarity_batch_size], dtype=np.float32)
        norms = np.linalg.norm(batch, axis=1)
        if not np.isfinite(batch).all() or np.any(norms == 0):
            msg = "topic-search embeddings contain invalid or zero vectors"
            raise ValueError(msg)
        similarities = batch @ query_vector / norms
        for offset, raw_similarity in enumerate(similarities):
            index = start + offset
            similarity = max(-1.0, min(1.0, float(raw_similarity)))
            if similarity < config.minimum_similarity or (int(labels[index]) < 0 and not config.include_outliers):
                continue
            candidate = (similarity, -index)
            if len(heap) < config.maximum_matches:
                heapq.heappush(heap, candidate)
            elif candidate > heap[0]:
                heapq.heapreplace(heap, candidate)
    return [(-negative_index, similarity) for similarity, negative_index in sorted(heap, reverse=True)]


def _safe_video_url(record: CorpusRecord) -> str:
    if not _YOUTUBE_VIDEO_ID.fullmatch(record.video_id):
        return ""
    return f"https://www.youtube.com/watch?v={record.video_id}"


def _load_evidence(
    artifacts: AnalysisArtifacts,
    matches: list[tuple[int, float]],
    signal_config: ProblemSignalConfig,
) -> list[TopicSearchEvidence]:
    selected = dict(matches)
    topics = {topic.topic_id: topic.name for topic in artifacts.topics}
    evidence: list[TopicSearchEvidence] = []
    rows = 0
    with artifacts.corpus_path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            index = rows
            rows += 1
            if index not in selected:
                continue
            record = CorpusRecord.model_validate_json(line)
            label = int(artifacts.labels[index])
            if label >= 0 and label not in topics:
                msg = f"topic-search assignment refers to unknown topic: {label}"
                raise ValueError(msg)
            evidence.append(
                TopicSearchEvidence(
                    record_index=index,
                    topic_id=label if label >= 0 else None,
                    topic_name=topics.get(label),
                    semantic_similarity=selected[index],
                    cluster_confidence=float(artifacts.confidence[index]),
                    problem_signals=match_problem_signals(record.clean_text, signal_config),
                    text=record.text,
                    text_kind=record.text_kind.value,
                    published_at=record.published_at,
                    video_title=record.video_title,
                    video_channel=record.video_channel,
                    video_url=_safe_video_url(record),
                )
            )
    if rows != artifacts.summary.records or rows != len(artifacts.labels):
        msg = "topic-search corpus and assignments are not row-aligned"
        raise ValueError(msg)
    if len(evidence) != len(matches):
        msg = "topic-search could not resolve every selected corpus row"
        raise ValueError(msg)
    if _sha256(artifacts.corpus_path) != artifacts.corpus.corpus_sha256:
        msg = "topic-search corpus changed while evidence was being loaded"
        raise ValueError(msg)
    return evidence


def _group_evidence(evidence: list[TopicSearchEvidence], limit: int) -> tuple[TopicSearchGroup, ...]:
    grouped: dict[int | None, list[TopicSearchEvidence]] = defaultdict(list)
    for item in evidence:
        grouped[item.topic_id].append(item)
    results = []
    for topic_id, rows in grouped.items():
        ranked = sorted(
            rows,
            key=lambda item: (not bool(item.problem_signals), -item.semantic_similarity, item.record_index),
        )
        problem_records = sum(bool(item.problem_signals) for item in rows)
        results.append(
            TopicSearchGroup(
                topic_id=topic_id,
                topic_name=rows[0].topic_name or "Outliers",
                relevant_records=len(rows),
                problem_records=problem_records,
                problem_share=problem_records / len(rows),
                unique_videos=len({item.video_url for item in rows if item.video_url}),
                maximum_similarity=max(item.semantic_similarity for item in rows),
                mean_similarity=sum(item.semantic_similarity for item in rows) / len(rows),
                evidence=tuple(ranked[:limit]),
            )
        )
    return tuple(
        sorted(
            results,
            key=lambda group: (-group.relevant_records, group.topic_id is None, group.topic_id or 0),
        )
    )


def search_user_topic(
    artifacts: AnalysisArtifacts,
    query: str,
    *,
    config: TopicSearchConfig | None = None,
    encoder: TextEncoder | None = None,
    searched_at: datetime | None = None,
) -> TopicSearchResult:
    """Find a relevant subcorpus and transparent problem evidence for a user theme."""
    normalized_query = " ".join(query.split())
    if not normalized_query or len(normalized_query) > _MAX_QUERY_LENGTH:
        msg = f"topic-search query must contain 1 to {_MAX_QUERY_LENGTH} characters"
        raise ValueError(msg)
    active_config = config or TopicSearchConfig()
    active_time = searched_at or datetime.now(UTC)
    if active_time.tzinfo is None:
        msg = "topic-search searched_at must be timezone-aware"
        raise ValueError(msg)
    matrix_path, manifest = _embedding_contract(artifacts)
    active_encoder = encoder or SentenceTransformerEncoder(_query_encoder_config(manifest))
    vector = _query_vector(active_encoder, normalized_query, artifacts.corpus.dimensions)
    matches = _top_matches(matrix_path, vector, artifacts.labels, active_config)
    if _sha256(matrix_path) != artifacts.corpus.final_embeddings_sha256:
        msg = "topic-search embeddings changed while similarities were being calculated"
        raise ValueError(msg)
    evidence = _load_evidence(artifacts, matches, active_config.problem_signals)
    groups = _group_evidence(evidence, active_config.evidence_per_topic)
    return TopicSearchResult(
        run_id=artifacts.run_id,
        pipeline_manifest_sha256=artifacts.pipeline_manifest_sha256,
        corpus_sha256=artifacts.corpus.corpus_sha256,
        embeddings_sha256=artifacts.corpus.final_embeddings_sha256,
        query=normalized_query,
        config=active_config,
        model_name=manifest.model_name,
        model_revision=manifest.model_revision,
        corpus_records=artifacts.summary.records,
        relevant_records=len(evidence),
        problem_records=sum(bool(item.problem_signals) for item in evidence),
        groups=groups,
        searched_at=active_time,
    )


def write_topic_search_result(
    result: TopicSearchResult,
    output_dir: Path,
    source_run_dir: Path,
    *,
    overwrite: bool = False,
) -> TopicSearchExportManifest:
    """Atomically persist restricted query results outside the immutable run."""
    target = output_dir.resolve()
    if output_dir.is_symlink() or target.is_relative_to(source_run_dir.resolve()):
        msg = "topic-search output must be a safe directory outside the immutable run"
        raise ValueError(msg)
    result_path = target / "topic-search-result.json"
    manifest_path = target / "topic-search-manifest.json"
    if any(path.is_symlink() for path in (result_path, manifest_path)):
        msg = "topic-search output files must not be symbolic links"
        raise ValueError(msg)
    if not overwrite and (result_path.exists() or manifest_path.exists()):
        msg = f"topic-search output already exists: {target}"
        raise FileExistsError(msg)
    target.mkdir(parents=True, exist_ok=True)
    result_tmp = result_path.with_name(f".{result_path.name}.tmp")
    manifest_tmp = manifest_path.with_name(f".{manifest_path.name}.tmp")
    if result_tmp.is_symlink() or manifest_tmp.is_symlink():
        msg = "topic-search temporary files must not be symbolic links"
        raise ValueError(msg)
    result_tmp.write_text(f"{result.model_dump_json(indent=2)}\n", encoding="utf-8")
    result_tmp.replace(result_path)
    manifest = TopicSearchExportManifest(
        run_id=result.run_id,
        pipeline_manifest_sha256=result.pipeline_manifest_sha256,
        query_sha256=hashlib.sha256(result.query.encode()).hexdigest(),
        result_path=str(result_path),
        result_sha256=_sha256(result_path),
        created_at=datetime.now(UTC),
    )
    manifest_tmp.write_text(f"{manifest.model_dump_json(indent=2)}\n", encoding="utf-8")
    manifest_tmp.replace(manifest_path)
    return manifest
