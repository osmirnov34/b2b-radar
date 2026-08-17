"""Conservative embedding-based reassignment of confident HDBSCAN outliers."""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.ml.clustering import ClusteringManifest
from src.ml.corpus import CorpusManifest, CorpusRecord
from src.ml.schemas import TextKind
from src.ml.topic_representation import TopicRepresentation, TopicRepresentationManifest

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

_EXPECTED_MATRIX_DIMENSIONS = 2


class _ReassignmentModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class OutlierReassignmentConfig(_ReassignmentModel):
    schema_version: int = 1
    enabled: bool = True
    similarity_threshold: float = Field(default=0.85, ge=0, le=1)
    single_topic_similarity_threshold: float = Field(default=0.9, ge=0, le=1)
    margin_threshold: float = Field(default=0.05, ge=0, le=2)
    minimum_topic_size: int = Field(default=20, ge=1)
    minimum_topic_mean_probability: float = Field(default=0.5, ge=0, le=1)
    centroid_member_minimum_probability: float = Field(default=0.7, ge=0, le=1)
    minimum_centroid_members: int = Field(default=10, ge=1)
    minimum_centroid_cohesion: float = Field(default=0.4, ge=-1, le=1)
    maximum_centroid_similarity: float = Field(default=0.95, ge=-1, le=1)
    maximum_reassigned_per_topic: int | None = Field(default=10_000, ge=1)
    maximum_topic_expansion_ratio: float = Field(default=0.25, ge=0)
    maximum_global_reassignment_share: float = Field(default=0.25, ge=0, le=1)
    block_size: int = Field(default=4096, ge=1)
    high_reassignment_share_warning: float = Field(default=0.2, ge=0, le=1)
    dominant_reassignment_topic_warning: float = Field(default=0.5, ge=0, le=1)
    low_margin_warning: float = Field(default=0.08, ge=0, le=2)
    random_seed: int = 42

    @model_validator(mode="after")
    def validate_thresholds(self) -> OutlierReassignmentConfig:
        if self.single_topic_similarity_threshold < self.similarity_threshold:
            msg = "single-topic similarity threshold cannot be less than the normal threshold"
            raise ValueError(msg)
        return self


class OutlierDecisionReason(StrEnum):
    REASSIGNED = "reassigned"
    DISABLED = "disabled"
    NO_ELIGIBLE_TOPIC = "no_eligible_topic"
    BELOW_SIMILARITY = "below_similarity"
    INSUFFICIENT_MARGIN = "insufficient_margin"
    TOPIC_LIMIT = "topic_limit"
    GLOBAL_LIMIT = "global_limit"


class OutlierDecision(_ReassignmentModel):
    record_index: int = Field(ge=0)
    final_label: int = Field(ge=-1)
    best_topic: int | None = Field(default=None, ge=0)
    best_similarity: float | None = Field(default=None, ge=-1, le=1)
    second_topic: int | None = Field(default=None, ge=0)
    second_similarity: float | None = Field(default=None, ge=-1, le=1)
    margin: float | None = Field(default=None, ge=0, le=2)
    reassigned: bool
    reason: OutlierDecisionReason
    config_schema_version: int = 1


class FinalClusterSummary(_ReassignmentModel):
    topic_id: int = Field(ge=0)
    original_records: int = Field(ge=1)
    reassigned_outliers: int = Field(ge=0)
    final_records: int = Field(ge=1)
    expansion_share: float = Field(ge=0)
    mean_reassignment_similarity: float | None = Field(default=None, ge=-1, le=1)
    minimum_reassignment_similarity: float | None = Field(default=None, ge=-1, le=1)
    comments: int = Field(ge=0)
    replies: int = Field(ge=0)
    languages: dict[str, int]
    unique_videos: int = Field(ge=0)


class OutlierReassignmentMetrics(_ReassignmentModel):
    records: int = Field(ge=0)
    topics: int = Field(ge=0)
    eligible_topics: int = Field(ge=0)
    original_outliers: int = Field(ge=0)
    reassigned_outliers: int = Field(ge=0)
    reassigned_share: float = Field(ge=0, le=1)
    remaining_outliers: int = Field(ge=0)
    largest_topic_reassignment_share: float = Field(ge=0, le=1)
    mean_accepted_similarity: float | None = Field(default=None, ge=-1, le=1)
    minimum_accepted_similarity: float | None = Field(default=None, ge=-1, le=1)
    mean_accepted_margin: float | None = Field(default=None, ge=0, le=2)
    rejection_reasons: dict[OutlierDecisionReason, int]


class OutlierReassignmentManifest(_ReassignmentModel):
    schema_version: int = 1
    corpus_manifest_path: str
    corpus_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    clustering_manifest_path: str
    clustering_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    topic_manifest_path: str
    topic_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    embeddings_path: str
    embeddings_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_labels_path: str
    source_labels_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_probabilities_path: str
    source_probabilities_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    topic_representations_path: str
    topic_representations_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    config: OutlierReassignmentConfig
    eligible_topics: list[int]
    ineligible_topics: dict[int, str]
    final_labels_path: str
    final_labels_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    final_confidence_path: str
    final_confidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    decisions_path: str
    decisions_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    summary_path: str
    summary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    original_confidence_source: str = "hdbscan_probability"
    reassigned_confidence_source: str = "embedding_cosine_similarity"
    metrics: OutlierReassignmentMetrics
    warnings: list[str]
    created_at: datetime


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_contracts(
    embeddings_path: Path,
    labels_path: Path,
    probabilities_path: Path,
    corpus_manifest_path: Path,
    clustering_manifest_path: Path,
    topic_manifest_path: Path,
) -> tuple[CorpusManifest, ClusteringManifest, TopicRepresentationManifest]:
    corpus = CorpusManifest.model_validate_json(corpus_manifest_path.read_text(encoding="utf-8"))
    clustering = ClusteringManifest.model_validate_json(clustering_manifest_path.read_text(encoding="utf-8"))
    topics = TopicRepresentationManifest.model_validate_json(topic_manifest_path.read_text(encoding="utf-8"))
    checks = (
        (_sha256_file(embeddings_path), corpus.final_embeddings_sha256, "embeddings"),
        (_sha256_file(labels_path), clustering.labels_sha256, "cluster labels"),
        (_sha256_file(probabilities_path), clustering.probabilities_sha256, "cluster probabilities"),
        (_sha256_file(corpus_manifest_path), clustering.corpus_manifest_sha256, "corpus manifest"),
        (_sha256_file(clustering_manifest_path), topics.clustering_manifest_sha256, "clustering manifest"),
    )
    for actual, expected, label in checks:
        if actual != expected:
            msg = f"{label} checksum does not match its downstream manifest"
            raise ValueError(msg)
    if topics.embeddings_sha256 != corpus.final_embeddings_sha256 or topics.labels_sha256 != clustering.labels_sha256:
        msg = "topic manifest refers to different embeddings or labels"
        raise ValueError(msg)
    if topics.omitted_topics:
        msg = "outlier reassignment requires a complete topic representation; limited topic runs are not accepted"
        raise ValueError(msg)
    return corpus, clustering, topics


def _load_topic_representations(path: Path, manifest: TopicRepresentationManifest) -> list[TopicRepresentation]:
    if not path.is_file() or _sha256_file(path) != manifest.representations_sha256:
        msg = "topic representations are missing or do not match topic manifest"
        raise ValueError(msg)
    items: list[TopicRepresentation] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                items.append(TopicRepresentation.model_validate_json(line))
            except ValueError as exc:
                msg = f"invalid topic representation at line {line_number}: {exc}"
                raise ValueError(msg) from exc
    if [item.topic_id for item in items] != list(range(manifest.topics)):
        msg = "topic representations must use all contiguous normalized topic IDs"
        raise ValueError(msg)
    return items


def _load_arrays(
    embeddings_path: Path,
    labels_path: Path,
    probabilities_path: Path,
    records: int,
    dimensions: int,
) -> tuple[NDArray[np.float32], NDArray[np.int64], NDArray[np.float32]]:
    import numpy as np

    embeddings = np.load(embeddings_path, mmap_mode="r", allow_pickle=False)
    labels = np.load(labels_path, mmap_mode="r", allow_pickle=False)
    probabilities = np.load(probabilities_path, mmap_mode="r", allow_pickle=False)
    invalid_embeddings = (
        embeddings.shape[0] < records
        or embeddings.ndim != _EXPECTED_MATRIX_DIMENSIONS
        or embeddings.shape[1] != dimensions
    )
    if invalid_embeddings:
        msg = "embedding matrix shape does not match reassignment inputs"
        raise ValueError(msg)
    if embeddings.dtype != np.float32:
        msg = f"embedding matrix must use float32, got {embeddings.dtype}"
        raise ValueError(msg)
    if labels.shape != (records,) or labels.dtype.kind not in "iu":
        msg = f"cluster labels must be an integer vector with {records} rows"
        raise ValueError(msg)
    if probabilities.shape != (records,):
        msg = f"cluster probabilities must contain {records} rows"
        raise ValueError(msg)
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        msg = "cluster probabilities must be finite values in [0, 1]"
        raise ValueError(msg)
    return embeddings, np.asarray(labels, dtype=np.int64), np.asarray(probabilities, dtype=np.float32)


def _normalized_chunk(vectors: NDArray[np.floating]) -> NDArray[np.float32]:
    import numpy as np

    result = np.asarray(vectors, dtype=np.float32)
    if not np.isfinite(result).all():
        msg = "embeddings contain NaN or infinite values"
        raise ValueError(msg)
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    if np.any(norms == 0):
        msg = "embeddings contain zero vectors"
        raise ValueError(msg)
    return np.asarray(result / norms, dtype=np.float32)


def _build_centroids(
    embeddings: NDArray[np.float32],
    labels: NDArray[np.int64],
    probabilities: NDArray[np.float32],
    representations: list[TopicRepresentation],
    config: OutlierReassignmentConfig,
) -> tuple[dict[int, NDArray[np.float32]], dict[int, str], dict[int, float]]:
    import numpy as np

    eligible = {
        item.topic_id
        for item in representations
        if item.records >= config.minimum_topic_size
        and item.mean_probability >= config.minimum_topic_mean_probability
        and item.keywords
    }
    ineligible = {
        item.topic_id: (
            "topic_too_small"
            if item.records < config.minimum_topic_size
            else "low_topic_probability"
            if item.mean_probability < config.minimum_topic_mean_probability
            else "empty_topic_representation"
        )
        for item in representations
        if item.topic_id not in eligible
    }
    dimensions = embeddings.shape[1]
    sums = {topic_id: np.zeros(dimensions, dtype=np.float64) for topic_id in eligible}
    counts: Counter[int] = Counter()
    for start in range(0, len(labels), config.block_size):
        stop = min(len(labels), start + config.block_size)
        vectors = _normalized_chunk(embeddings[start:stop])
        chunk_labels = labels[start:stop]
        chunk_probabilities = probabilities[start:stop]
        for topic_id in tuple(eligible):
            mask = (chunk_labels == topic_id) & (
                chunk_probabilities >= config.centroid_member_minimum_probability
            )
            if np.any(mask):
                sums[topic_id] += np.sum(vectors[mask], axis=0)
                counts[topic_id] += int(np.count_nonzero(mask))
    centroids: dict[int, NDArray[np.float32]] = {}
    for topic_id in sorted(eligible):
        norm = float(np.linalg.norm(sums[topic_id]))
        if counts[topic_id] < config.minimum_centroid_members:
            ineligible[topic_id] = "insufficient_centroid_members"
        elif norm == 0:
            ineligible[topic_id] = "zero_centroid"
        else:
            centroids[topic_id] = np.asarray(sums[topic_id] / norm, dtype=np.float32)
    cohesion_sums: dict[int, float] = defaultdict(float)
    for start in range(0, len(labels), config.block_size):
        stop = min(len(labels), start + config.block_size)
        vectors = _normalized_chunk(embeddings[start:stop])
        chunk_labels = labels[start:stop]
        chunk_probabilities = probabilities[start:stop]
        for topic_id, centroid in centroids.items():
            mask = (chunk_labels == topic_id) & (
                chunk_probabilities >= config.centroid_member_minimum_probability
            )
            if np.any(mask):
                cohesion_sums[topic_id] += float(np.sum(vectors[mask] @ centroid))
    cohesions = {topic_id: cohesion_sums[topic_id] / counts[topic_id] for topic_id in centroids}
    for topic_id, cohesion in tuple(cohesions.items()):
        if cohesion < config.minimum_centroid_cohesion:
            centroids.pop(topic_id)
            ineligible[topic_id] = "low_centroid_cohesion"
    topic_ids = sorted(centroids)
    if len(topic_ids) > 1:
        matrix = np.asarray([centroids[topic_id] for topic_id in topic_ids], dtype=np.float32)
        similarities = matrix @ matrix.T
        ambiguous: set[int] = set()
        for left in range(len(topic_ids)):
            for right in range(left + 1, len(topic_ids)):
                if similarities[left, right] > config.maximum_centroid_similarity:
                    ambiguous.update((topic_ids[left], topic_ids[right]))
        for topic_id in ambiguous:
            centroids.pop(topic_id)
            ineligible[topic_id] = "ambiguous_centroid"
    return centroids, ineligible, cohesions


@dataclass(slots=True)
class _Candidate:
    record_index: int
    best_topic: int | None = None
    best_similarity: float | None = None
    second_topic: int | None = None
    second_similarity: float | None = None
    margin: float | None = None
    reason: OutlierDecisionReason = OutlierDecisionReason.NO_ELIGIBLE_TOPIC
    accepted: bool = False


def _score_outliers(
    embeddings: NDArray[np.float32],
    outlier_indices: NDArray[np.int64],
    centroids: dict[int, NDArray[np.float32]],
    config: OutlierReassignmentConfig,
) -> list[_Candidate]:
    import numpy as np

    if not config.enabled:
        return [
            _Candidate(record_index=int(index), reason=OutlierDecisionReason.DISABLED)
            for index in outlier_indices
        ]
    if not centroids:
        return [_Candidate(record_index=int(index)) for index in outlier_indices]
    topic_ids = np.asarray(sorted(centroids), dtype=np.int64)
    centroid_matrix = np.asarray([centroids[int(topic_id)] for topic_id in topic_ids], dtype=np.float32)
    candidates: list[_Candidate] = []
    for start in range(0, len(outlier_indices), config.block_size):
        block_indices = outlier_indices[start : start + config.block_size]
        vectors = _normalized_chunk(embeddings[block_indices])
        similarities = np.clip(vectors @ centroid_matrix.T, -1.0, 1.0)
        for row, record_index in enumerate(block_indices):
            order = np.argsort(similarities[row], kind="stable")[::-1]
            best_column = int(order[0])
            best_similarity = float(similarities[row, best_column])
            best_topic = int(topic_ids[best_column])
            candidate = _Candidate(
                record_index=int(record_index),
                best_topic=best_topic,
                best_similarity=best_similarity,
            )
            if len(topic_ids) == 1:
                if best_similarity >= config.single_topic_similarity_threshold:
                    candidate.accepted = True
                    candidate.reason = OutlierDecisionReason.REASSIGNED
                else:
                    candidate.reason = OutlierDecisionReason.BELOW_SIMILARITY
            else:
                second_column = int(order[1])
                second_similarity = float(similarities[row, second_column])
                margin = best_similarity - second_similarity
                candidate.second_topic = int(topic_ids[second_column])
                candidate.second_similarity = second_similarity
                candidate.margin = margin
                if best_similarity < config.similarity_threshold:
                    candidate.reason = OutlierDecisionReason.BELOW_SIMILARITY
                elif margin < config.margin_threshold:
                    candidate.reason = OutlierDecisionReason.INSUFFICIENT_MARGIN
                else:
                    candidate.accepted = True
                    candidate.reason = OutlierDecisionReason.REASSIGNED
            candidates.append(candidate)
    return candidates


def _apply_limits(
    candidates: list[_Candidate],
    original_sizes: dict[int, int],
    outliers: int,
    config: OutlierReassignmentConfig,
) -> None:
    accepted = [candidate for candidate in candidates if candidate.accepted]
    accepted.sort(
        key=lambda item: (
            -(item.best_similarity if item.best_similarity is not None else -1),
            -(item.margin if item.margin is not None else 0),
            item.record_index,
        ),
    )
    global_limit = math.ceil(outliers * config.maximum_global_reassignment_share)
    topic_counts: Counter[int] = Counter()
    total = 0
    for candidate in accepted:
        assert candidate.best_topic is not None  # noqa: S101 - accepted candidates always have a best topic
        expansion_limit = math.ceil(original_sizes[candidate.best_topic] * config.maximum_topic_expansion_ratio)
        if config.maximum_reassigned_per_topic is not None:
            expansion_limit = min(expansion_limit, config.maximum_reassigned_per_topic)
        if total >= global_limit:
            candidate.accepted = False
            candidate.reason = OutlierDecisionReason.GLOBAL_LIMIT
        elif topic_counts[candidate.best_topic] >= expansion_limit:
            candidate.accepted = False
            candidate.reason = OutlierDecisionReason.TOPIC_LIMIT
        else:
            topic_counts[candidate.best_topic] += 1
            total += 1


@dataclass(slots=True)
class _FinalSummaryState:
    original: int = 0
    reassigned_similarities: list[float] = field(default_factory=list)
    kinds: Counter[TextKind] = field(default_factory=Counter)
    languages: Counter[str] = field(default_factory=Counter)
    videos: set[str] = field(default_factory=set)


def _final_summaries(
    corpus_path: Path,
    source_labels: NDArray[np.int64],
    final_labels: NDArray[np.int64],
    accepted_similarities: dict[int, float],
    topics: int,
) -> list[FinalClusterSummary]:
    states = [_FinalSummaryState() for _ in range(topics)]
    count = 0
    with corpus_path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            if count >= len(final_labels):
                break
            record = CorpusRecord.model_validate_json(line)
            topic_id = int(final_labels[count])
            if topic_id >= 0:
                state = states[topic_id]
                if source_labels[count] >= 0:
                    state.original += 1
                else:
                    state.reassigned_similarities.append(accepted_similarities[count])
                state.kinds[record.text_kind] += 1
                state.languages[record.detected_language] += 1
                if record.video_id:
                    state.videos.add(record.video_id)
            count += 1
    if count != len(final_labels):
        msg = f"corpus supplied {count} records for {len(final_labels)} final labels"
        raise ValueError(msg)
    summaries = []
    for topic_id, state in enumerate(states):
        reassigned = len(state.reassigned_similarities)
        summaries.append(
            FinalClusterSummary(
                topic_id=topic_id,
                original_records=state.original,
                reassigned_outliers=reassigned,
                final_records=state.original + reassigned,
                expansion_share=reassigned / state.original,
                mean_reassignment_similarity=(
                    sum(state.reassigned_similarities) / reassigned if reassigned else None
                ),
                minimum_reassignment_similarity=(
                    min(state.reassigned_similarities) if reassigned else None
                ),
                comments=state.kinds[TextKind.COMMENT],
                replies=state.kinds[TextKind.REPLY],
                languages=dict(state.languages),
                unique_videos=len(state.videos),
            ),
        )
    return summaries


def _save_numpy(path: Path, values: NDArray[np.generic]) -> None:
    import numpy as np

    with path.open("wb") as target:
        np.save(target, values, allow_pickle=False)


def _ensure_source_labels_unchanged(path: Path, expected_sha256: str) -> None:
    if _sha256_file(path) != expected_sha256:
        msg = "source cluster labels changed during outlier reassignment"
        raise ValueError(msg)


def reassign_outliers(
    embeddings_path: Path,
    labels_path: Path,
    probabilities_path: Path,
    corpus_path: Path,
    corpus_manifest_path: Path,
    clustering_manifest_path: Path,
    topic_manifest_path: Path,
    output_dir: Path,
    *,
    config: OutlierReassignmentConfig | None = None,
    force: bool = False,
) -> OutlierReassignmentManifest:
    """Reassign only high-similarity, high-margin outliers and preserve uncertainty."""
    import numpy as np

    active_config = config or OutlierReassignmentConfig()
    corpus, clustering, topic_manifest = _validate_contracts(
        embeddings_path,
        labels_path,
        probabilities_path,
        corpus_manifest_path,
        clustering_manifest_path,
        topic_manifest_path,
    )
    if _sha256_file(corpus_path) != corpus.corpus_sha256:
        msg = "final corpus checksum does not match corpus manifest"
        raise ValueError(msg)
    representations_path = Path(topic_manifest.representations_path)
    representations = _load_topic_representations(representations_path, topic_manifest)
    records = clustering.output_records
    embeddings, labels, probabilities = _load_arrays(
        embeddings_path,
        labels_path,
        probabilities_path,
        records,
        corpus.dimensions,
    )
    topic_ids = sorted({int(label) for label in labels if label >= 0})
    if topic_ids != list(range(topic_manifest.topics)):
        msg = "cluster labels and topic manifest disagree on normalized topics"
        raise ValueError(msg)

    output_dir.mkdir(parents=True, exist_ok=True)
    final_labels_path = output_dir / "final-cluster-labels.npy"
    final_confidence_path = output_dir / "final-cluster-confidence.npy"
    decisions_path = output_dir / "outlier-decisions.jsonl"
    summary_path = output_dir / "final-cluster-summary.jsonl"
    manifest_path = output_dir / "outlier-reassignment-manifest.json"
    report_path = output_dir / "outlier-reassignment-report.md"
    finals = (
        final_labels_path,
        final_confidence_path,
        decisions_path,
        summary_path,
        manifest_path,
        report_path,
    )
    existing = next((path for path in finals if path.exists()), None)
    if existing is not None and not force:
        msg = f"refusing to overwrite existing outlier-reassignment artifact: {existing}"
        raise FileExistsError(msg)

    centroids, ineligible, _cohesions = _build_centroids(
        embeddings,
        labels,
        probabilities,
        representations,
        active_config,
    )
    outlier_indices = np.flatnonzero(labels == -1).astype(np.int64)
    candidates = _score_outliers(embeddings, outlier_indices, centroids, active_config)
    original_sizes = Counter(int(label) for label in labels if label >= 0)
    _apply_limits(candidates, dict(original_sizes), len(outlier_indices), active_config)
    final_labels = np.asarray(labels.copy(), dtype=np.int64)
    final_confidence = np.asarray(probabilities.copy(), dtype=np.float32)
    accepted_similarities: dict[int, float] = {}
    for candidate in candidates:
        if candidate.accepted:
            assert candidate.best_topic is not None  # noqa: S101 - accepted candidate invariant
            assert candidate.best_similarity is not None  # noqa: S101 - accepted candidate invariant
            final_labels[candidate.record_index] = candidate.best_topic
            final_confidence[candidate.record_index] = candidate.best_similarity
            accepted_similarities[candidate.record_index] = candidate.best_similarity
    decisions = [
        OutlierDecision(
            record_index=candidate.record_index,
            final_label=int(final_labels[candidate.record_index]),
            best_topic=candidate.best_topic,
            best_similarity=candidate.best_similarity,
            second_topic=candidate.second_topic,
            second_similarity=candidate.second_similarity,
            margin=candidate.margin,
            reassigned=candidate.accepted,
            reason=candidate.reason,
        )
        for candidate in sorted(candidates, key=lambda item: item.record_index)
    ]
    summaries = _final_summaries(
        corpus_path,
        labels,
        final_labels,
        accepted_similarities,
        topic_manifest.topics,
    )
    accepted = [decision for decision in decisions if decision.reassigned]
    accepted_sim_values = [decision.best_similarity for decision in accepted if decision.best_similarity is not None]
    accepted_margins = [decision.margin for decision in accepted if decision.margin is not None]
    rejection_reasons = Counter(decision.reason for decision in decisions if not decision.reassigned)
    accepted_by_topic = Counter(decision.best_topic for decision in accepted)
    metrics = OutlierReassignmentMetrics(
        records=records,
        topics=topic_manifest.topics,
        eligible_topics=len(centroids),
        original_outliers=len(outlier_indices),
        reassigned_outliers=len(accepted),
        reassigned_share=len(accepted) / len(outlier_indices) if len(outlier_indices) else 0,
        remaining_outliers=len(outlier_indices) - len(accepted),
        largest_topic_reassignment_share=(
            max(accepted_by_topic.values(), default=0) / len(accepted) if accepted else 0
        ),
        mean_accepted_similarity=(
            sum(accepted_sim_values) / len(accepted_sim_values) if accepted_sim_values else None
        ),
        minimum_accepted_similarity=min(accepted_sim_values, default=None),
        mean_accepted_margin=sum(accepted_margins) / len(accepted_margins) if accepted_margins else None,
        rejection_reasons=dict(rejection_reasons),
    )
    warnings = []
    if metrics.reassigned_share > active_config.high_reassignment_share_warning:
        warnings.append(f"high reassignment share: {metrics.reassigned_share:.1%}")
    if metrics.mean_accepted_margin is not None and metrics.mean_accepted_margin < active_config.low_margin_warning:
        warnings.append(f"low mean accepted margin: {metrics.mean_accepted_margin:.4f}")
    if metrics.largest_topic_reassignment_share > active_config.dominant_reassignment_topic_warning:
        warnings.append(
            f"one topic received {metrics.largest_topic_reassignment_share:.1%} of reassigned outliers",
        )
    if ineligible:
        warnings.append(f"topics excluded from reassignment: {len(ineligible)}")

    temporary = {path: path.with_name(f".{path.name}.tmp") for path in finals}
    for path in temporary.values():
        path.unlink(missing_ok=True)
    try:
        _save_numpy(temporary[final_labels_path], final_labels)
        _save_numpy(temporary[final_confidence_path], final_confidence)
        with temporary[decisions_path].open("w", encoding="utf-8") as target:
            for decision in decisions:
                target.write(f"{decision.model_dump_json()}\n")
        with temporary[summary_path].open("w", encoding="utf-8") as target:
            for summary in summaries:
                target.write(f"{summary.model_dump_json()}\n")
        _ensure_source_labels_unchanged(labels_path, clustering.labels_sha256)
        manifest = OutlierReassignmentManifest(
            corpus_manifest_path=str(corpus_manifest_path),
            corpus_manifest_sha256=_sha256_file(corpus_manifest_path),
            clustering_manifest_path=str(clustering_manifest_path),
            clustering_manifest_sha256=_sha256_file(clustering_manifest_path),
            topic_manifest_path=str(topic_manifest_path),
            topic_manifest_sha256=_sha256_file(topic_manifest_path),
            embeddings_path=str(embeddings_path),
            embeddings_sha256=corpus.final_embeddings_sha256,
            source_labels_path=str(labels_path),
            source_labels_sha256=clustering.labels_sha256,
            source_probabilities_path=str(probabilities_path),
            source_probabilities_sha256=clustering.probabilities_sha256,
            topic_representations_path=str(representations_path),
            topic_representations_sha256=topic_manifest.representations_sha256,
            config=active_config,
            eligible_topics=sorted(centroids),
            ineligible_topics=ineligible,
            final_labels_path=str(final_labels_path),
            final_labels_sha256=_sha256_file(temporary[final_labels_path]),
            final_confidence_path=str(final_confidence_path),
            final_confidence_sha256=_sha256_file(temporary[final_confidence_path]),
            decisions_path=str(decisions_path),
            decisions_sha256=_sha256_file(temporary[decisions_path]),
            summary_path=str(summary_path),
            summary_sha256=_sha256_file(temporary[summary_path]),
            metrics=metrics,
            warnings=warnings,
            created_at=datetime.now(UTC),
        )
        temporary[manifest_path].write_text(f"{manifest.model_dump_json(indent=2)}\n", encoding="utf-8")
        temporary[report_path].write_text(_reassignment_report(manifest), encoding="utf-8")
        publication_order = (
            final_labels_path,
            final_confidence_path,
            decisions_path,
            summary_path,
            report_path,
            manifest_path,
        )
        for final in publication_order:
            temporary[final].replace(final)
    except Exception:
        for path in temporary.values():
            path.unlink(missing_ok=True)
        raise
    return manifest


def _reassignment_report(manifest: OutlierReassignmentManifest) -> str:
    metrics = manifest.metrics
    reasons = "\n".join(f"- `{reason}`: {count}" for reason, count in sorted(metrics.rejection_reasons.items()))
    warnings = "\n".join(f"- {warning}" for warning in manifest.warnings) or "- None."
    return f"""# Outlier reassignment

- Records: {metrics.records}
- Topics: {metrics.topics}
- Eligible topics: {metrics.eligible_topics}
- Original outliers: {metrics.original_outliers}
- Reassigned outliers: {metrics.reassigned_outliers} ({metrics.reassigned_share:.1%})
- Remaining outliers: {metrics.remaining_outliers}
- Largest topic share of reassigned outliers: {metrics.largest_topic_reassignment_share:.1%}
- Mean accepted similarity: {metrics.mean_accepted_similarity}
- Minimum accepted similarity: {metrics.minimum_accepted_similarity}
- Mean accepted margin: {metrics.mean_accepted_margin}

## Rejection reasons

{reasons or '- None.'}

## Warnings

{warnings}

Original HDBSCAN labels remain unchanged. Final confidence means HDBSCAN probability for original members and embedding
cosine similarity for reassigned outliers. Decision and summary artifacts contain indexes and aggregates only, never
comment text or source identifiers.
"""
