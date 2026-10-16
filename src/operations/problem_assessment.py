"""Deterministic hybrid problem assessment for canonical customer feedback."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from src.analytics import (
    AssessmentMethod,
    EvidenceSpan,
    FeedbackMLAdaptationManifest,
    FeedbackPreparationManifest,
    FeedbackProblemAssessment,
    FeedbackRecord,
    FeedbackStance,
    FeedbackTopicAssignment,
    ProblemAssessmentArtifactManifest,
    ProblemDetectionRules,
    ProblemTaxonomy,
    iter_feedback_records,
    load_problem_detection_rules,
    load_problem_taxonomy,
    validate_problem_detection_rules,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@dataclass(slots=True)
class _AssessmentCounts:
    feedback: int = 0
    assessments: int = 0
    assessed_feedback: int = 0
    methods: Counter[AssessmentMethod] = field(default_factory=Counter)
    stances: Counter[FeedbackStance] = field(default_factory=Counter)
    categories: Counter[str] = field(default_factory=Counter)


@dataclass(frozen=True, slots=True)
class _AssessmentContext:
    taxonomy: ProblemTaxonomy
    rules: ProblemDetectionRules
    topics: dict[str, FeedbackTopicAssignment]
    topic_revision: str | None
    assessed_at: datetime


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_topic_assignments(
    path: Path | None,
    manifest_path: Path | None,
) -> tuple[dict[str, FeedbackTopicAssignment], str | None]:
    if path is None and manifest_path is None:
        return {}, None
    if path is None or manifest_path is None:
        msg = "topic assignments and their manifest must be configured together"
        raise ValueError(msg)
    manifest = FeedbackMLAdaptationManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    checksum = _sha256_file(path)
    if checksum != manifest.output_sha256:
        msg = "topic assignments do not match their adaptation manifest"
        raise ValueError(msg)
    assignments: dict[str, FeedbackTopicAssignment] = {}
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                assignment = FeedbackTopicAssignment.model_validate_json(line)
            except ValueError as exc:
                msg = f"invalid topic assignment at line {line_number}"
                raise ValueError(msg) from exc
            if assignment.feedback_id in assignments:
                msg = f"duplicate topic assignment for feedback: {assignment.feedback_id!r}"
                raise ValueError(msg)
            assignments[assignment.feedback_id] = assignment
    if len(assignments) != manifest.matched_records:
        msg = "topic assignment count does not match adaptation manifest"
        raise ValueError(msg)
    return assignments, checksum


def _evidence_spans(text: str, terms: tuple[str, ...]) -> tuple[EvidenceSpan, ...]:
    folded = text.casefold()
    candidates: list[tuple[int, int]] = []
    for term in terms:
        start = 0
        while (found := folded.find(term, start)) >= 0:
            candidates.append((found, found + len(term)))
            start = found + len(term)
    selected: list[tuple[int, int]] = []
    for start, end in sorted(candidates, key=lambda span: (span[0], -(span[1] - span[0]))):
        if selected and start < selected[-1][1]:
            continue
        selected.append((start, end))
    return tuple(EvidenceSpan(start=start, end=end) for start, end in selected)


def _term_positions(text: str, terms: tuple[str, ...]) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    folded = text.casefold()
    for term in terms:
        start = 0
        while (found := folded.find(term, start)) >= 0:
            positions.append((found, found + len(term)))
            start = found + len(term)
    return positions


def _has_unshadowed_match(matches: list[tuple[int, int]], opposing: list[tuple[int, int]]) -> bool:
    return any(
        not any(
            left < opposing_right
            and opposing_left < right
            and opposing_right - opposing_left > right - left
            for opposing_left, opposing_right in opposing
        )
        for left, right in matches
    )


def _stance(record: FeedbackRecord, rules: ProblemDetectionRules) -> tuple[FeedbackStance, float]:
    praise_positions = _term_positions(record.text, rules.praise_terms)
    complaint_positions = _term_positions(record.text, rules.complaint_terms)
    praise = _has_unshadowed_match(praise_positions, complaint_positions)
    complaint = _has_unshadowed_match(complaint_positions, praise_positions)
    if complaint and praise:
        return FeedbackStance.MIXED, 0.80
    rating_complaint = record.rating <= rules.complaint_rating_max
    rating_praise = record.rating >= rules.praise_rating_min
    if complaint:
        return FeedbackStance.COMPLAINT, 0.95 if rating_complaint else 0.80
    if praise:
        return FeedbackStance.PRAISE, 0.95 if rating_praise else 0.80
    if rating_complaint:
        return FeedbackStance.COMPLAINT, 0.65
    if rating_praise:
        return FeedbackStance.PRAISE, 0.65
    return FeedbackStance.UNCLEAR, 0.50


def _assess_record(
    record: FeedbackRecord,
    context: _AssessmentContext,
) -> Iterator[FeedbackProblemAssessment]:
    stance, stance_probability = _stance(record, context.rules)
    topic = context.topics.get(record.feedback_id)
    topic_name = (topic.topic_name or "").casefold() if topic is not None else ""
    for category_rule in context.rules.categories:
        spans = _evidence_spans(record.text, category_rule.evidence_terms)
        if not spans:
            continue
        topic_support = bool(topic_name) and any(term in topic_name for term in category_rule.evidence_terms)
        method = AssessmentMethod.HYBRID if topic_support else AssessmentMethod.RULE
        probability = min(0.65 + 0.05 * (len(spans) - 1) + (0.15 if topic_support else 0), 0.95)
        yield FeedbackProblemAssessment(
            feedback_id=record.feedback_id,
            taxonomy_id=context.taxonomy.taxonomy_id,
            taxonomy_version=context.taxonomy.version,
            category_id=category_rule.category_id,
            stance=stance,
            category_probability=probability,
            stance_probability=stance_probability,
            method=method,
            evidence_spans=spans,
            model_name="existing-ml-topic-assignment" if topic_support else None,
            model_revision=context.topic_revision if topic_support else None,
            assessed_at=context.assessed_at,
        )


def _write_assessments(
    feedback_path: Path,
    output_path: Path,
    context: _AssessmentContext,
) -> _AssessmentCounts:
    counts = _AssessmentCounts()
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        with temporary.open("x", encoding="utf-8") as target:
            for record in iter_feedback_records(feedback_path):
                counts.feedback += 1
                record_assessments = 0
                for assessment in _assess_record(record, context):
                    target.write(f"{assessment.model_dump_json()}\n")
                    counts.assessments += 1
                    record_assessments += 1
                    counts.methods[assessment.method] += 1
                    counts.stances[assessment.stance] += 1
                    counts.categories[assessment.category_id] += 1
                counts.assessed_feedback += record_assessments > 0
        temporary.replace(output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return counts


def assess_feedback_problems(  # noqa: PLR0913 -- explicit paths make artifact lineage auditable.
    feedback_path: Path,
    feedback_manifest_path: Path,
    taxonomy_path: Path,
    rules_path: Path,
    output_path: Path,
    *,
    topic_assignments_path: Path | None = None,
    topic_manifest_path: Path | None = None,
) -> ProblemAssessmentArtifactManifest:
    """Generate unreviewed evidence-bound problem and strength candidates."""
    manifest_path = output_path.with_suffix(f"{output_path.suffix}.manifest.json")
    existing = next((path for path in (output_path, manifest_path) if path.exists()), None)
    if existing is not None:
        msg = f"refusing to overwrite existing problem assessment artifact: {existing}"
        raise FileExistsError(msg)
    feedback_manifest = FeedbackPreparationManifest.model_validate_json(
        feedback_manifest_path.read_text(encoding="utf-8"),
    )
    if _sha256_file(feedback_path) != feedback_manifest.output_sha256:
        msg = "canonical feedback does not match its preparation manifest"
        raise ValueError(msg)
    taxonomy = load_problem_taxonomy(taxonomy_path)
    rules = validate_problem_detection_rules(load_problem_detection_rules(rules_path), taxonomy)
    topics, topics_sha256 = _load_topic_assignments(topic_assignments_path, topic_manifest_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(UTC)
    context = _AssessmentContext(
        taxonomy=taxonomy,
        rules=rules,
        topics=topics,
        topic_revision=topics_sha256,
        assessed_at=generated_at,
    )
    counts = _write_assessments(feedback_path, output_path, context)
    if counts.feedback != feedback_manifest.records:
        output_path.unlink(missing_ok=True)
        msg = "canonical feedback count does not match its preparation manifest"
        raise ValueError(msg)
    manifest = ProblemAssessmentArtifactManifest(
        feedback_path=str(feedback_path),
        feedback_sha256=feedback_manifest.output_sha256,
        taxonomy_path=str(taxonomy_path),
        taxonomy_sha256=_sha256_file(taxonomy_path),
        taxonomy_id=taxonomy.taxonomy_id,
        taxonomy_version=taxonomy.version,
        rules_path=str(rules_path),
        rules_sha256=_sha256_file(rules_path),
        ruleset_id=rules.ruleset_id,
        ruleset_version=rules.version,
        topic_assignments_path=str(topic_assignments_path) if topic_assignments_path is not None else None,
        topic_assignments_sha256=topics_sha256,
        output_path=str(output_path),
        output_sha256=_sha256_file(output_path),
        generated_at=generated_at,
        feedback_records=counts.feedback,
        assessed_feedback_records=counts.assessed_feedback,
        assessments=counts.assessments,
        rule_assessments=counts.methods[AssessmentMethod.RULE],
        hybrid_assessments=counts.methods[AssessmentMethod.HYBRID],
        complaint_assessments=counts.stances[FeedbackStance.COMPLAINT],
        praise_assessments=counts.stances[FeedbackStance.PRAISE],
        neutral_assessments=counts.stances[FeedbackStance.NEUTRAL],
        mixed_assessments=counts.stances[FeedbackStance.MIXED],
        unclear_assessments=counts.stances[FeedbackStance.UNCLEAR],
        category_counts=dict(counts.categories),
    )
    try:
        with manifest_path.open("x", encoding="utf-8") as target:
            target.write(f"{manifest.model_dump_json(indent=2)}\n")
    except Exception:
        output_path.unlink(missing_ok=True)
        raise
    return manifest
