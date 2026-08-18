"""Checksum-bound exports of fixed topic-analysis results."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, cast

from pydantic import BaseModel, ConfigDict, Field

from src.ml.clustering import ClusteringManifest
from src.ml.corpus import CorpusManifest, CorpusRecord
from src.ml.evaluation import EvaluationManifest, EvaluationMetrics, EvaluationStatus
from src.ml.outlier_reassignment import OutlierDecision, OutlierReassignmentManifest
from src.ml.topic_representation import TopicRepresentation, TopicRepresentationManifest

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

_EMAIL_PATTERN = re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b")
_PHONE_PATTERN = re.compile(r"(?<!\d)(?:\+?\d[\s().-]?){7,}\d(?!\d)")


class _ExportModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AssignmentSource(StrEnum):
    HDBSCAN = "hdbscan"
    REASSIGNMENT = "reassignment"
    OUTLIER = "outlier"


class ExportConfig(_ExportModel):
    schema_version: int = 1
    include_research_text: bool = False
    require_final_evaluation: bool = False
    include_topic_keywords: bool = True
    maximum_keywords_per_topic: int = Field(default=10, ge=0, le=100)


class ExportedTopic(_ExportModel):
    topic_id: int = Field(ge=0)
    name: str
    keywords: list[str]
    original_records: int = Field(ge=1)
    final_records: int = Field(ge=1)
    reassigned_records: int = Field(ge=0)
    original_languages: dict[str, int]
    original_unique_videos: int = Field(ge=0)
    mean_probability: float = Field(ge=0, le=1)


class ExportedAssignment(_ExportModel):
    corpus_id: str = Field(pattern=r"^corpus:[0-9a-f]{64}$")
    topic_id: int | None = Field(default=None, ge=0)
    source: AssignmentSource
    confidence: float = Field(ge=0, le=1)
    outlier_reason: str | None = None


class ResearchAssignment(ExportedAssignment):
    record_id: str
    text: str
    clean_text: str
    text_kind: str
    parent_record_id: str | None = None
    detected_language: str
    video_id: str
    author: str
    search_query: str


class ExportQuality(_ExportModel):
    evaluation_status: str
    preliminary: bool
    records: int = Field(ge=0)
    topics: int = Field(ge=0)
    final_outlier_share: float = Field(ge=0, le=1)
    mean_bootstrap_ari: float | None = Field(default=None, ge=-1, le=1)
    manual_annotations: int = Field(ge=0)
    warnings: list[str]


class ExportManifest(_ExportModel):
    schema_version: int = 1
    config: ExportConfig
    corpus_manifest_path: str
    corpus_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    clustering_manifest_path: str
    clustering_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    topic_manifest_path: str
    topic_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reassignment_manifest_path: str
    reassignment_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_manifest_path: str
    evaluation_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    topics_path: str
    topics_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    assignments_path: str
    assignments_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    quality_path: str
    quality_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    research_assignments_path: str | None = None
    research_assignments_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    records: int = Field(ge=0)
    topics: int = Field(ge=0)
    outliers: int = Field(ge=0)
    created_at: datetime


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_manifests(
    corpus_manifest_path: Path,
    clustering_manifest_path: Path,
    topic_manifest_path: Path,
    reassignment_manifest_path: Path,
    evaluation_manifest_path: Path,
) -> tuple[
    CorpusManifest,
    ClusteringManifest,
    TopicRepresentationManifest,
    OutlierReassignmentManifest,
    EvaluationManifest,
    EvaluationMetrics,
]:
    corpus = CorpusManifest.model_validate_json(corpus_manifest_path.read_text(encoding="utf-8"))
    clustering = ClusteringManifest.model_validate_json(clustering_manifest_path.read_text(encoding="utf-8"))
    topics = TopicRepresentationManifest.model_validate_json(topic_manifest_path.read_text(encoding="utf-8"))
    reassignment = OutlierReassignmentManifest.model_validate_json(
        reassignment_manifest_path.read_text(encoding="utf-8"),
    )
    evaluation = EvaluationManifest.model_validate_json(evaluation_manifest_path.read_text(encoding="utf-8"))
    checks = (
        (_sha256_file(corpus_manifest_path), clustering.corpus_manifest_sha256, "corpus manifest"),
        (_sha256_file(clustering_manifest_path), topics.clustering_manifest_sha256, "clustering manifest"),
        (_sha256_file(topic_manifest_path), reassignment.topic_manifest_sha256, "topic manifest"),
        (_sha256_file(corpus_manifest_path), evaluation.corpus_manifest_sha256, "evaluation corpus manifest"),
        (
            _sha256_file(clustering_manifest_path),
            evaluation.clustering_manifest_sha256,
            "evaluation clustering manifest",
        ),
        (_sha256_file(topic_manifest_path), evaluation.topic_manifest_sha256, "evaluation topic manifest"),
        (
            _sha256_file(reassignment_manifest_path),
            evaluation.reassignment_manifest_sha256,
            "evaluation reassignment manifest",
        ),
    )
    for actual, expected, label in checks:
        if actual != expected:
            msg = f"{label} checksum does not match its downstream manifest"
            raise ValueError(msg)
    metrics_path = Path(evaluation.metrics_path)
    if not metrics_path.is_file() or _sha256_file(metrics_path) != evaluation.metrics_sha256:
        msg = "evaluation metrics are missing or do not match evaluation manifest"
        raise ValueError(msg)
    metrics = EvaluationMetrics.model_validate_json(metrics_path.read_text(encoding="utf-8"))
    records = corpus.stats.output_records
    if records != reassignment.metrics.records or records != metrics.records:
        msg = "record counts differ across corpus, reassignment, and evaluation manifests"
        raise ValueError(msg)
    if topics.omitted_topics:
        msg = "export requires complete topic representations"
        raise ValueError(msg)
    return corpus, clustering, topics, reassignment, evaluation, metrics


def _load_labels(path: Path, expected_hash: str, records: int) -> NDArray[np.int64]:
    import numpy as np

    if not path.is_file() or _sha256_file(path) != expected_hash:
        msg = f"labels are missing or do not match manifest: {path}"
        raise ValueError(msg)
    labels = np.load(path, mmap_mode="r", allow_pickle=False)
    if labels.shape != (records,) or labels.dtype.kind not in "iu":
        msg = f"invalid labels shape or dtype: {path}"
        raise ValueError(msg)
    return cast("NDArray[np.int64]", labels)


def _load_confidence(path: Path, expected_hash: str, records: int) -> NDArray[np.float32]:
    import numpy as np

    if not path.is_file() or _sha256_file(path) != expected_hash:
        msg = "final confidence is missing or does not match reassignment manifest"
        raise ValueError(msg)
    confidence = np.load(path, mmap_mode="r", allow_pickle=False)
    invalid = (
        confidence.shape != (records,)
        or not np.isfinite(confidence).all()
        or np.any((confidence < 0) | (confidence > 1))
    )
    if invalid:
        msg = "final confidence must be an aligned finite vector in [0, 1]"
        raise ValueError(msg)
    return cast("NDArray[np.float32]", confidence)


def _load_topics(manifest: TopicRepresentationManifest) -> list[TopicRepresentation]:
    path = Path(manifest.representations_path)
    if not path.is_file() or _sha256_file(path) != manifest.representations_sha256:
        msg = "topic representations are missing or do not match topic manifest"
        raise ValueError(msg)
    topics = [
        TopicRepresentation.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if [topic.topic_id for topic in topics] != list(range(manifest.topics)):
        msg = "topic representations must contain contiguous normalized topic IDs"
        raise ValueError(msg)
    return topics


def _outlier_reasons(
    manifest: OutlierReassignmentManifest,
    source_labels: NDArray[np.int64],
    final_labels: NDArray[np.int64],
) -> dict[int, str]:
    path = Path(manifest.decisions_path)
    if not path.is_file() or _sha256_file(path) != manifest.decisions_sha256:
        msg = "outlier decisions are missing or do not match reassignment manifest"
        raise ValueError(msg)
    reasons: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            decision = OutlierDecision.model_validate_json(line)
            if decision.record_index in reasons:
                msg = "outlier decisions contain duplicate record indices"
                raise ValueError(msg)
            reasons[decision.record_index] = decision.reason.value
            outside = decision.record_index >= len(final_labels)
            mismatched = not outside and decision.final_label != int(final_labels[decision.record_index])
            if outside or mismatched:
                msg = "outlier decision does not match the final labels array"
                raise ValueError(msg)
    expected = {index for index, label in enumerate(source_labels) if label == -1}
    if set(reasons) != expected:
        msg = "outlier decisions do not cover exactly the original outliers"
        raise ValueError(msg)
    return reasons


def _safe_keywords(topic: TopicRepresentation, config: ExportConfig) -> list[str]:
    if not config.include_topic_keywords:
        return []
    return [
        keyword.term
        for keyword in topic.keywords
        if not _EMAIL_PATTERN.search(keyword.term) and not _PHONE_PATTERN.search(keyword.term)
    ][: config.maximum_keywords_per_topic]


def _safe_topic_name(topic: TopicRepresentation, keywords: list[str]) -> str:
    if not _EMAIL_PATTERN.search(topic.name) and not _PHONE_PATTERN.search(topic.name):
        return topic.name
    return " / ".join(keywords[:3]) or f"Topic {topic.topic_id}"


def _assignment(
    record: CorpusRecord,
    index: int,
    source_label: int,
    final_label: int,
    confidence: float,
    reasons: dict[int, str],
) -> ExportedAssignment:
    if final_label == -1:
        source = AssignmentSource.OUTLIER
        reason = reasons.get(index, "unassigned")
    elif source_label == -1:
        source = AssignmentSource.REASSIGNMENT
        reason = None
    else:
        source = AssignmentSource.HDBSCAN
        reason = None
    return ExportedAssignment(
        corpus_id=record.corpus_id,
        topic_id=None if final_label == -1 else final_label,
        source=source,
        confidence=confidence,
        outlier_reason=reason,
    )


def export_topic_results(
    corpus_manifest_path: Path,
    clustering_manifest_path: Path,
    topic_manifest_path: Path,
    reassignment_manifest_path: Path,
    evaluation_manifest_path: Path,
    output_dir: Path,
    *,
    config: ExportConfig | None = None,
    force: bool = False,
) -> ExportManifest:
    """Create public JSONL exports and an optional sensitive research export."""
    import numpy as np

    active_config = config or ExportConfig()
    corpus, _clustering, topics_manifest, reassignment, evaluation, metrics = _validated_manifests(
        corpus_manifest_path,
        clustering_manifest_path,
        topic_manifest_path,
        reassignment_manifest_path,
        evaluation_manifest_path,
    )
    if active_config.require_final_evaluation and (
        metrics.preliminary or metrics.status != EvaluationStatus.PASS
    ):
        msg = "final export requires a passing completed manual review and validation"
        raise ValueError(msg)
    records = corpus.stats.output_records
    source_labels = _load_labels(Path(reassignment.source_labels_path), reassignment.source_labels_sha256, records)
    final_labels = _load_labels(Path(reassignment.final_labels_path), reassignment.final_labels_sha256, records)
    confidence = _load_confidence(
        Path(reassignment.final_confidence_path),
        reassignment.final_confidence_sha256,
        records,
    )
    if np.any(final_labels < -1) or np.any(final_labels >= topics_manifest.topics):
        msg = "final labels refer to unknown topics"
        raise ValueError(msg)
    topics = _load_topics(topics_manifest)
    reasons = _outlier_reasons(reassignment, source_labels, final_labels)
    final_counts = Counter(int(label) for label in final_labels if label >= 0)
    reassigned_counts = Counter(
        int(final_labels[index])
        for index in range(records)
        if source_labels[index] == -1 and final_labels[index] >= 0
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    topics_path = output_dir / "topics.jsonl"
    assignments_path = output_dir / "assignments.jsonl"
    quality_path = output_dir / "quality.json"
    research_path = output_dir / "research-assignments.jsonl"
    manifest_path = output_dir / "export-manifest.json"
    finals = [topics_path, assignments_path, quality_path, manifest_path]
    if active_config.include_research_text:
        finals.append(research_path)
    elif research_path.exists():
        msg = "output directory contains a sensitive research export; use a separate public output directory"
        raise ValueError(msg)
    existing = next((path for path in finals if path.exists()), None)
    if existing is not None and not force:
        msg = f"refusing to overwrite existing export artifact: {existing}"
        raise FileExistsError(msg)
    temporary = {path: path.with_name(f".{path.name}.tmp") for path in finals}
    for path in temporary.values():
        path.unlink(missing_ok=True)
    try:
        with temporary[topics_path].open("w", encoding="utf-8") as target:
            for topic in topics:
                safe_keywords = _safe_keywords(topic, active_config)
                item = ExportedTopic(
                    topic_id=topic.topic_id,
                    name=_safe_topic_name(topic, safe_keywords),
                    keywords=safe_keywords,
                    original_records=topic.records,
                    final_records=final_counts[topic.topic_id],
                    reassigned_records=reassigned_counts[topic.topic_id],
                    original_languages=topic.languages,
                    original_unique_videos=topic.unique_videos,
                    mean_probability=topic.mean_probability,
                )
                target.write(f"{item.model_dump_json()}\n")
        corpus_path = Path(corpus.corpus_path)
        if not corpus_path.is_file() or _sha256_file(corpus_path) != corpus.corpus_sha256:
            msg = "corpus is missing or does not match corpus manifest"
            raise ValueError(msg)
        research_target = (
            temporary[research_path].open("w", encoding="utf-8") if active_config.include_research_text else None
        )
        try:
            with corpus_path.open(encoding="utf-8") as source, temporary[assignments_path].open(
                "w",
                encoding="utf-8",
            ) as public_target:
                index = 0
                for line in source:
                    if not line.strip():
                        continue
                    if index >= records:
                        msg = "corpus contains more records than its manifest"
                        raise ValueError(msg)
                    record = CorpusRecord.model_validate_json(line)
                    assignment = _assignment(
                        record,
                        index,
                        int(source_labels[index]),
                        int(final_labels[index]),
                        float(confidence[index]),
                        reasons,
                    )
                    public_target.write(f"{assignment.model_dump_json()}\n")
                    if research_target is not None:
                        research = ResearchAssignment(
                            **assignment.model_dump(),
                            record_id=record.record_id,
                            text=record.text,
                            clean_text=record.clean_text,
                            text_kind=record.text_kind.value,
                            parent_record_id=record.parent_record_id,
                            detected_language=record.detected_language,
                            video_id=record.video_id,
                            author=record.author,
                            search_query=record.search_query,
                        )
                        research_target.write(f"{research.model_dump_json()}\n")
                    index += 1
                if index != records:
                    msg = f"corpus contains {index} records; expected {records}"
                    raise ValueError(msg)
        finally:
            if research_target is not None:
                research_target.close()
        quality = ExportQuality(
            evaluation_status=metrics.status.value,
            preliminary=metrics.preliminary,
            records=records,
            topics=topics_manifest.topics,
            final_outlier_share=metrics.final_outlier_share,
            mean_bootstrap_ari=metrics.mean_bootstrap_ari,
            manual_annotations=metrics.manual.annotations,
            warnings=evaluation.warnings,
        )
        temporary[quality_path].write_text(f"{quality.model_dump_json(indent=2)}\n", encoding="utf-8")
        manifest = ExportManifest(
            config=active_config,
            corpus_manifest_path=str(corpus_manifest_path),
            corpus_manifest_sha256=_sha256_file(corpus_manifest_path),
            clustering_manifest_path=str(clustering_manifest_path),
            clustering_manifest_sha256=_sha256_file(clustering_manifest_path),
            topic_manifest_path=str(topic_manifest_path),
            topic_manifest_sha256=_sha256_file(topic_manifest_path),
            reassignment_manifest_path=str(reassignment_manifest_path),
            reassignment_manifest_sha256=_sha256_file(reassignment_manifest_path),
            evaluation_manifest_path=str(evaluation_manifest_path),
            evaluation_manifest_sha256=_sha256_file(evaluation_manifest_path),
            topics_path=str(topics_path),
            topics_sha256=_sha256_file(temporary[topics_path]),
            assignments_path=str(assignments_path),
            assignments_sha256=_sha256_file(temporary[assignments_path]),
            quality_path=str(quality_path),
            quality_sha256=_sha256_file(temporary[quality_path]),
            research_assignments_path=str(research_path) if active_config.include_research_text else None,
            research_assignments_sha256=(
                _sha256_file(temporary[research_path]) if active_config.include_research_text else None
            ),
            records=records,
            topics=topics_manifest.topics,
            outliers=int(np.count_nonzero(final_labels == -1)),
            created_at=datetime.now(UTC),
        )
        temporary[manifest_path].write_text(f"{manifest.model_dump_json(indent=2)}\n", encoding="utf-8")
        for final in (topics_path, assignments_path, quality_path):
            temporary[final].replace(final)
        if active_config.include_research_text:
            temporary[research_path].replace(research_path)
        temporary[manifest_path].replace(manifest_path)
    except Exception:
        for path in temporary.values():
            path.unlink(missing_ok=True)
        raise
    return manifest
