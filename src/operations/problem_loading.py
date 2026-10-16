"""Verified transactional import of problem assessments into PostgreSQL."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import ValidationError
from sqlalchemy import func, insert, select
from sqlalchemy.exc import SQLAlchemyError

from src.analytics import (
    AssessmentMethod,
    FeedbackProblemAssessment,
    FeedbackStance,
    ProblemAssessmentArtifactManifest,
)
from src.infrastructure.analytics_db import (
    AnalyticsDatasetModel,
    AnalyticsFeedbackModel,
    AnalyticsProblemAssessmentModel,
    AnalyticsProblemRunModel,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncSession


class ProblemLoadError(RuntimeError):
    """Assessment import failure that never includes private feedback text."""


@dataclass(frozen=True, slots=True)
class VerifiedProblemArtifact:
    """Fully parsed, aggregate-verified assessment artifact."""

    manifest: ProblemAssessmentArtifactManifest
    assessments: tuple[FeedbackProblemAssessment, ...]


@dataclass(frozen=True, slots=True)
class ProblemLoadResult:
    """Outcome of an idempotent problem-assessment import."""

    run_id: int
    dataset_id: int
    assessments: int
    already_loaded: bool


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(path: Path) -> ProblemAssessmentArtifactManifest:
    try:
        return ProblemAssessmentArtifactManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValidationError) as exc:
        msg = "problem manifest is unreadable or invalid"
        raise ProblemLoadError(msg) from exc


def _iter_assessments(path: Path) -> Iterator[FeedbackProblemAssessment]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                yield FeedbackProblemAssessment.model_validate_json(line)
            except ValueError as exc:
                msg = f"invalid problem assessment at line {line_number}"
                raise ProblemLoadError(msg) from exc


def verify_problem_artifact(artifact_path: Path, manifest_path: Path) -> VerifiedProblemArtifact:
    """Verify checksum, unique keys, taxonomy bindings, and aggregate counts."""
    if not artifact_path.is_file():
        msg = f"problem assessment artifact not found: {artifact_path}"
        raise ProblemLoadError(msg)
    manifest = _load_manifest(manifest_path)
    if _sha256(artifact_path) != manifest.output_sha256:
        msg = "problem assessment artifact checksum does not match manifest"
        raise ProblemLoadError(msg)
    assessments = tuple(_iter_assessments(artifact_path))
    keys: set[tuple[str, str]] = set()
    feedback_ids: set[str] = set()
    methods: Counter[AssessmentMethod] = Counter()
    stances: Counter[FeedbackStance] = Counter()
    categories: Counter[str] = Counter()
    for assessment in assessments:
        key = (assessment.feedback_id, assessment.category_id)
        if key in keys:
            msg = "problem assessment artifact contains duplicate feedback/category keys"
            raise ProblemLoadError(msg)
        keys.add(key)
        feedback_ids.add(assessment.feedback_id)
        methods[assessment.method] += 1
        stances[assessment.stance] += 1
        categories[assessment.category_id] += 1
        if (assessment.taxonomy_id, assessment.taxonomy_version) != (
            manifest.taxonomy_id,
            manifest.taxonomy_version,
        ):
            msg = "problem assessment taxonomy does not match manifest"
            raise ProblemLoadError(msg)
    expected_methods = {
        AssessmentMethod.RULE: manifest.rule_assessments,
        AssessmentMethod.HYBRID: manifest.hybrid_assessments,
    }
    expected_stances = {
        FeedbackStance.COMPLAINT: manifest.complaint_assessments,
        FeedbackStance.PRAISE: manifest.praise_assessments,
        FeedbackStance.NEUTRAL: manifest.neutral_assessments,
        FeedbackStance.MIXED: manifest.mixed_assessments,
        FeedbackStance.UNCLEAR: manifest.unclear_assessments,
    }
    if len(assessments) != manifest.assessments or len(feedback_ids) != manifest.assessed_feedback_records:
        msg = "problem assessment record counts do not match manifest"
        raise ProblemLoadError(msg)
    if any(methods[method] != count for method, count in expected_methods.items()):
        msg = "problem assessment method counts do not match manifest"
        raise ProblemLoadError(msg)
    if any(stances[stance] != count for stance, count in expected_stances.items()):
        msg = "problem assessment stance counts do not match manifest"
        raise ProblemLoadError(msg)
    if dict(categories) != manifest.category_counts:
        msg = "problem assessment category counts do not match manifest"
        raise ProblemLoadError(msg)
    return VerifiedProblemArtifact(manifest=manifest, assessments=assessments)


def _assessment_batches(
    verified: VerifiedProblemArtifact,
    run_id: int,
    dataset_id: int,
    batch_size: int,
) -> Iterator[list[dict[str, object]]]:
    batch: list[dict[str, object]] = []
    for assessment in verified.assessments:
        batch.append(
            {
                "run_id": run_id,
                "dataset_id": dataset_id,
                "feedback_id": assessment.feedback_id,
                "category_id": assessment.category_id,
                "stance": assessment.stance.value,
                "category_probability": assessment.category_probability,
                "stance_probability": assessment.stance_probability,
                "method": assessment.method.value,
                "evidence_spans": json.dumps(
                    [span.model_dump() for span in assessment.evidence_spans],
                    separators=(",", ":"),
                ),
                "model_name": assessment.model_name,
                "model_revision": assessment.model_revision,
                "review_status": assessment.review_status.value,
                "reviewer_id": assessment.reviewer_id,
                "assessed_at": assessment.assessed_at,
            },
        )
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


async def _validate_feedback_bindings(
    session: AsyncSession,
    dataset_id: int,
    assessments: tuple[FeedbackProblemAssessment, ...],
) -> None:
    rows = await session.execute(
        select(AnalyticsFeedbackModel.feedback_id, func.length(AnalyticsFeedbackModel.text)).where(
            AnalyticsFeedbackModel.dataset_id == dataset_id,
        ),
    )
    text_lengths: dict[str, int] = {
        str(feedback_id): int(text_length)
        for feedback_id, text_length in rows.tuples()
    }
    for assessment in assessments:
        length = text_lengths.get(assessment.feedback_id)
        if length is None:
            msg = "problem assessment references feedback outside the selected dataset"
            raise ProblemLoadError(msg)
        if any(span.end > length for span in assessment.evidence_spans):
            msg = "problem assessment evidence lies outside source feedback text"
            raise ProblemLoadError(msg)


async def load_problem_assessments(
    session: AsyncSession,
    artifact_path: Path,
    manifest_path: Path,
    *,
    batch_size: int = 1_000,
) -> ProblemLoadResult:
    """Import one verified assessment run in a single owned transaction."""
    if batch_size < 1:
        msg = "batch_size must be at least 1"
        raise ProblemLoadError(msg)
    if session.in_transaction():
        msg = "loader requires a session without an active transaction"
        raise ProblemLoadError(msg)
    verified = verify_problem_artifact(artifact_path, manifest_path)
    manifest = verified.manifest
    try:
        async with session.begin():
            existing = await session.scalar(
                select(AnalyticsProblemRunModel).where(
                    AnalyticsProblemRunModel.artifact_sha256 == manifest.output_sha256,
                ),
            )
            if existing is not None:
                count = await session.scalar(
                    select(func.count())
                    .select_from(AnalyticsProblemAssessmentModel)
                    .where(AnalyticsProblemAssessmentModel.run_id == existing.id),
                )
                if count != manifest.assessments:
                    msg = "existing problem run is incomplete or inconsistent with manifest"
                    raise ProblemLoadError(msg)
                return ProblemLoadResult(
                    run_id=existing.id,
                    dataset_id=existing.dataset_id,
                    assessments=manifest.assessments,
                    already_loaded=True,
                )
            dataset = await session.scalar(
                select(AnalyticsDatasetModel).where(
                    AnalyticsDatasetModel.artifact_sha256 == manifest.feedback_sha256,
                ),
            )
            if dataset is None:
                msg = "canonical feedback snapshot must be loaded before problem assessments"
                raise ProblemLoadError(msg)
            if dataset.record_count != manifest.feedback_records:
                msg = "problem manifest feedback count does not match stored dataset"
                raise ProblemLoadError(msg)
            await _validate_feedback_bindings(session, dataset.id, verified.assessments)
            run = AnalyticsProblemRunModel(
                dataset_id=dataset.id,
                artifact_sha256=manifest.output_sha256,
                taxonomy_id=manifest.taxonomy_id,
                taxonomy_version=manifest.taxonomy_version,
                ruleset_id=manifest.ruleset_id,
                ruleset_version=manifest.ruleset_version,
                topic_assignments_sha256=manifest.topic_assignments_sha256,
                generated_at=manifest.generated_at,
                feedback_count=manifest.feedback_records,
                assessed_feedback_count=manifest.assessed_feedback_records,
                assessment_count=manifest.assessments,
            )
            session.add(run)
            await session.flush()
            for batch in _assessment_batches(verified, run.id, dataset.id, batch_size):
                await session.execute(insert(AnalyticsProblemAssessmentModel), batch)
    except SQLAlchemyError:
        msg = "database rejected problem assessments; transaction rolled back"
        raise ProblemLoadError(msg) from None
    return ProblemLoadResult(
        run_id=run.id,
        dataset_id=dataset.id,
        assessments=manifest.assessments,
        already_loaded=False,
    )
