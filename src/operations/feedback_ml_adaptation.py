"""Join checksum-verified ML topic assignments to canonical feedback records."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from src.analytics import (
    FeedbackMLAdaptationManifest,
    FeedbackPreparationManifest,
    FeedbackRecord,
    FeedbackTopicAssignment,
    iter_feedback_records,
)
from src.ml.corpus import CorpusManifest, CorpusRecord
from src.ml.outlier_reassignment import OutlierDecision, OutlierReassignmentManifest
from src.ml.topic_representation import TopicRepresentation, TopicRepresentationManifest

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class _VerifiedMLInputs:
    corpus_manifest: CorpusManifest
    corpus_path: Path
    reassignment: OutlierReassignmentManifest
    topic_manifest_path: Path
    topics: dict[int, str]
    labels: NDArray[np.integer]
    confidence: NDArray[np.floating]
    decisions: list[OutlierDecision]


@dataclass(slots=True)
class _AdaptationCounts:
    corpus: int = 0
    assigned: int = 0
    outliers: int = 0
    reassigned: int = 0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_path(value: str, checksum: str, label: str) -> Path:
    path = Path(value)
    if not path.is_file() or _sha256_file(path) != checksum:
        msg = f"{label} is missing or does not match its manifest"
        raise ValueError(msg)
    return path


def _feedback_index(path: Path) -> tuple[dict[str, FeedbackRecord], int]:
    aliases: dict[str, FeedbackRecord] = {}
    records = 0
    for record in iter_feedback_records(path):
        records += 1
        for identifier in (record.feedback_id, record.source_record_id):
            existing = aliases.get(identifier)
            if existing is not None and existing.feedback_id != record.feedback_id:
                msg = f"ambiguous canonical feedback identifier: {identifier!r}"
                raise ValueError(msg)
            aliases[identifier] = record
    return aliases, records


def _iter_decisions(path: Path) -> Iterator[OutlierDecision]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                yield OutlierDecision.model_validate_json(line)
            except ValueError as exc:
                msg = f"invalid reassignment decision at line {line_number}"
                raise ValueError(msg) from exc


def _load_ml_inputs(corpus_manifest_path: Path, reassignment_manifest_path: Path) -> _VerifiedMLInputs:
    corpus_manifest = CorpusManifest.model_validate_json(corpus_manifest_path.read_text(encoding="utf-8"))
    corpus_path = _verified_path(corpus_manifest.corpus_path, corpus_manifest.corpus_sha256, "final corpus")
    reassignment = OutlierReassignmentManifest.model_validate_json(
        reassignment_manifest_path.read_text(encoding="utf-8"),
    )
    if reassignment.corpus_manifest_sha256 != _sha256_file(corpus_manifest_path):
        msg = "reassignment result belongs to a different corpus manifest"
        raise ValueError(msg)
    labels_path = _verified_path(reassignment.final_labels_path, reassignment.final_labels_sha256, "final labels")
    confidence_path = _verified_path(
        reassignment.final_confidence_path,
        reassignment.final_confidence_sha256,
        "final confidence",
    )
    decisions_path = _verified_path(reassignment.decisions_path, reassignment.decisions_sha256, "decisions")
    topic_manifest_path = Path(reassignment.topic_manifest_path)
    if _sha256_file(topic_manifest_path) != reassignment.topic_manifest_sha256:
        msg = "topic manifest does not match reassignment manifest"
        raise ValueError(msg)
    topic_manifest = TopicRepresentationManifest.model_validate_json(topic_manifest_path.read_text(encoding="utf-8"))
    topics_path = _verified_path(
        topic_manifest.representations_path,
        topic_manifest.representations_sha256,
        "topic representations",
    )
    topics = {
        topic.topic_id: topic.name
        for line in topics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for topic in (TopicRepresentation.model_validate_json(line),)
    }
    labels = np.load(labels_path, mmap_mode="r", allow_pickle=False)
    confidence = np.load(confidence_path, mmap_mode="r", allow_pickle=False)
    decisions = list(_iter_decisions(decisions_path))
    expected = corpus_manifest.stats.output_records
    if labels.shape != (expected,) or confidence.shape != (expected,) or len(decisions) != expected:
        msg = "corpus, labels, confidence, and decisions are not aligned"
        raise ValueError(msg)
    if not np.isfinite(confidence).all() or np.any((confidence < 0) | (confidence > 1)):
        msg = "final confidence must contain finite values in [0, 1]"
        raise ValueError(msg)
    return _VerifiedMLInputs(
        corpus_manifest=corpus_manifest,
        corpus_path=corpus_path,
        reassignment=reassignment,
        topic_manifest_path=topic_manifest_path,
        topics=topics,
        labels=labels,
        confidence=confidence,
        decisions=decisions,
    )


def _make_assignment(
    record: CorpusRecord,
    index: int,
    feedback_by_id: dict[str, FeedbackRecord],
    seen_feedback: set[str],
    inputs: _VerifiedMLInputs,
) -> FeedbackTopicAssignment:
    feedback = feedback_by_id.get(record.record_id)
    if feedback is None:
        msg = f"corpus record has no canonical feedback match: {record.record_id!r}"
        raise ValueError(msg)
    if feedback.feedback_id in seen_feedback:
        msg = f"canonical feedback occurs more than once in corpus: {feedback.feedback_id!r}"
        raise ValueError(msg)
    decision = inputs.decisions[index]
    label = int(inputs.labels[index])
    if decision.record_index != index or decision.final_label != label:
        msg = f"reassignment decision is misaligned at corpus index {index}"
        raise ValueError(msg)
    topic_name = inputs.topics.get(label) if label >= 0 else None
    if label >= 0 and topic_name is None:
        msg = f"final label has no topic representation: {label}"
        raise ValueError(msg)
    seen_feedback.add(feedback.feedback_id)
    return FeedbackTopicAssignment(
        **feedback.model_dump(exclude={"text"}),
        corpus_id=record.corpus_id,
        topic_id=label if label >= 0 else None,
        topic_name=topic_name,
        confidence=float(inputs.confidence[index]),
        reassigned=decision.reassigned,
    )


def _validate_corpus_count(actual: int, expected: int) -> None:
    if actual != expected:
        msg = f"corpus contains {actual} records; manifest declares {expected}"
        raise ValueError(msg)


def _write_assignments(
    output_path: Path,
    inputs: _VerifiedMLInputs,
    feedback_by_id: dict[str, FeedbackRecord],
) -> tuple[_AdaptationCounts, set[str]]:
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    temporary.unlink(missing_ok=True)
    counts = _AdaptationCounts()
    seen_feedback: set[str] = set()
    try:
        with inputs.corpus_path.open(encoding="utf-8") as corpus, temporary.open("x", encoding="utf-8") as target:
            for line in corpus:
                if not line.strip():
                    continue
                assignment = _make_assignment(
                    CorpusRecord.model_validate_json(line),
                    counts.corpus,
                    feedback_by_id,
                    seen_feedback,
                    inputs,
                )
                target.write(f"{assignment.model_dump_json()}\n")
                counts.corpus += 1
                counts.assigned += assignment.topic_id is not None
                counts.outliers += assignment.topic_id is None
                counts.reassigned += assignment.reassigned
        _validate_corpus_count(counts.corpus, inputs.corpus_manifest.stats.output_records)
        temporary.replace(output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return counts, seen_feedback


def adapt_feedback_ml_results(
    feedback_path: Path,
    feedback_manifest_path: Path,
    corpus_manifest_path: Path,
    reassignment_manifest_path: Path,
    output_path: Path,
) -> FeedbackMLAdaptationManifest:
    """Publish a source-neutral feedback/topic artifact without copying private text."""
    manifest_path = output_path.with_suffix(f"{output_path.suffix}.manifest.json")
    existing = next((path for path in (output_path, manifest_path) if path.exists()), None)
    if existing is not None:
        msg = f"refusing to overwrite existing ML adaptation artifact: {existing}"
        raise FileExistsError(msg)
    feedback_manifest = FeedbackPreparationManifest.model_validate_json(
        feedback_manifest_path.read_text(encoding="utf-8"),
    )
    if _sha256_file(feedback_path) != feedback_manifest.output_sha256:
        msg = "canonical feedback does not match its preparation manifest"
        raise ValueError(msg)
    inputs = _load_ml_inputs(corpus_manifest_path, reassignment_manifest_path)
    feedback_by_id, feedback_records = _feedback_index(feedback_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts, seen_feedback = _write_assignments(output_path, inputs, feedback_by_id)
    manifest = FeedbackMLAdaptationManifest(
        feedback_path=str(feedback_path),
        feedback_sha256=feedback_manifest.output_sha256,
        feedback_manifest_path=str(feedback_manifest_path),
        feedback_manifest_sha256=_sha256_file(feedback_manifest_path),
        corpus_path=str(inputs.corpus_path),
        corpus_sha256=inputs.corpus_manifest.corpus_sha256,
        reassignment_manifest_path=str(reassignment_manifest_path),
        reassignment_manifest_sha256=_sha256_file(reassignment_manifest_path),
        topic_manifest_path=str(inputs.topic_manifest_path),
        topic_manifest_sha256=inputs.reassignment.topic_manifest_sha256,
        output_path=str(output_path),
        output_sha256=_sha256_file(output_path),
        adapted_at=datetime.now(UTC),
        feedback_records=feedback_records,
        corpus_records=counts.corpus,
        matched_records=len(seen_feedback),
        unmatched_feedback_records=feedback_records - len(seen_feedback),
        assigned_records=counts.assigned,
        outlier_records=counts.outliers,
        reassigned_records=counts.reassigned,
        topics=len(inputs.topics),
    )
    try:
        with manifest_path.open("x", encoding="utf-8") as target:
            target.write(f"{manifest.model_dump_json(indent=2)}\n")
    except Exception:
        output_path.unlink(missing_ok=True)
        raise
    return manifest
