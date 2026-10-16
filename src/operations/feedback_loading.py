"""Verified transactional import of canonical feedback into PostgreSQL."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import ValidationError
from sqlalchemy import func, insert, select
from sqlalchemy.exc import SQLAlchemyError

from src.analytics import FeedbackPreparationManifest, FeedbackRecord, iter_feedback_records
from src.infrastructure.analytics_db import (
    AnalyticsDatasetModel,
    AnalyticsFeedbackModel,
    AnalyticsOrganizationModel,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncSession


class AnalyticsLoadError(RuntimeError):
    """Raised before or during an import without exposing private feedback."""


@dataclass(frozen=True)
class VerifiedFeedbackArtifact:
    """Aggregate preflight result safe to carry into a database transaction."""

    manifest: FeedbackPreparationManifest
    organizations: tuple[tuple[str, str, str], ...]


@dataclass(frozen=True)
class AnalyticsLoadResult:
    """Outcome of an idempotent snapshot load."""

    dataset_id: int
    records: int
    organizations: int
    already_loaded: bool


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(path: Path) -> FeedbackPreparationManifest:
    if not path.is_file():
        msg = f"manifest file not found: {path}"
        raise AnalyticsLoadError(msg)
    try:
        return FeedbackPreparationManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValidationError) as exc:
        msg = "manifest is unreadable or does not match FeedbackPreparationManifest"
        raise AnalyticsLoadError(msg) from exc


def verify_feedback_artifact(artifact_path: Path, manifest_path: Path) -> VerifiedFeedbackArtifact:
    """Verify bytes, record alignment, aggregates, and organization identity."""
    if not artifact_path.is_file():
        msg = f"feedback artifact not found: {artifact_path}"
        raise AnalyticsLoadError(msg)
    manifest = _load_manifest(manifest_path)
    if artifact_path.stat().st_size != manifest.output_size_bytes:
        msg = "feedback artifact size does not match manifest"
        raise AnalyticsLoadError(msg)
    if _sha256(artifact_path) != manifest.output_sha256:
        msg = "feedback artifact checksum does not match manifest"
        raise AnalyticsLoadError(msg)

    feedback_ids: set[str] = set()
    source_record_ids: set[str] = set()
    organizations: dict[str, tuple[str, str]] = {}
    ratings: Counter[int] = Counter()
    published_dates = []
    for record in iter_feedback_records(artifact_path):
        _observe_record(record, manifest, feedback_ids, source_record_ids, organizations)
        ratings[record.rating] += 1
        published_dates.append(record.published_on)

    if len(feedback_ids) != manifest.records:
        msg = "feedback record count does not match manifest"
        raise AnalyticsLoadError(msg)
    if len(organizations) != manifest.unique_organizations:
        msg = "organization count does not match manifest"
        raise AnalyticsLoadError(msg)
    if dict(sorted(ratings.items())) != manifest.rating_distribution:
        msg = "rating distribution does not match manifest"
        raise AnalyticsLoadError(msg)
    actual_min = min(published_dates, default=None)
    actual_max = max(published_dates, default=None)
    if (actual_min, actual_max) != (manifest.published_on_min, manifest.published_on_max):
        msg = "publication date range does not match manifest"
        raise AnalyticsLoadError(msg)
    return VerifiedFeedbackArtifact(
        manifest=manifest,
        organizations=tuple((identifier, *details) for identifier, details in sorted(organizations.items())),
    )


def _observe_record(
    record: FeedbackRecord,
    manifest: FeedbackPreparationManifest,
    feedback_ids: set[str],
    source_record_ids: set[str],
    organizations: dict[str, tuple[str, str]],
) -> None:
    if record.source != manifest.source:
        msg = "feedback source does not match manifest"
        raise AnalyticsLoadError(msg)
    if record.feedback_id in feedback_ids or record.source_record_id in source_record_ids:
        msg = "duplicate feedback identifier detected"
        raise AnalyticsLoadError(msg)
    feedback_ids.add(record.feedback_id)
    source_record_ids.add(record.source_record_id)
    identity = (record.organization_name, record.organization_url)
    existing = organizations.setdefault(record.organization_id, identity)
    if existing != identity:
        msg = "organization identity is inconsistent inside the snapshot"
        raise AnalyticsLoadError(msg)


def _feedback_batches(path: Path, dataset_id: int, batch_size: int) -> Iterator[list[dict[str, object]]]:
    batch: list[dict[str, object]] = []
    for record in iter_feedback_records(path):
        batch.append(
            {
                "dataset_id": dataset_id,
                "feedback_id": record.feedback_id,
                "source_record_id": record.source_record_id,
                "organization_id": record.organization_id,
                "text": record.text,
                "rating": record.rating,
                "published_on": record.published_on,
                "search_query": record.search_query,
            },
        )
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


async def load_feedback_snapshot(
    session: AsyncSession,
    artifact_path: Path,
    manifest_path: Path,
    *,
    batch_size: int = 1_000,
) -> AnalyticsLoadResult:
    """Preflight an artifact and import it in one owned database transaction."""
    if batch_size < 1:
        msg = "batch_size must be at least 1"
        raise AnalyticsLoadError(msg)
    if session.in_transaction():
        msg = "loader requires a session without an active transaction"
        raise AnalyticsLoadError(msg)
    verified = verify_feedback_artifact(artifact_path, manifest_path)
    manifest = verified.manifest

    try:
        async with session.begin():
            existing = await session.scalar(
                select(AnalyticsDatasetModel).where(AnalyticsDatasetModel.artifact_sha256 == manifest.output_sha256),
            )
            if existing is not None:
                return await _existing_result(session, existing, manifest)

            dataset = AnalyticsDatasetModel(
                source=manifest.source.value,
                manifest_schema_version=manifest.schema_version,
                source_sha256=manifest.source_sha256,
                artifact_sha256=manifest.output_sha256,
                prepared_at=manifest.prepared_at,
                record_count=manifest.records,
                organization_count=manifest.unique_organizations,
                published_on_min=manifest.published_on_min,
                published_on_max=manifest.published_on_max,
            )
            session.add(dataset)
            await session.flush()
            await session.execute(
                insert(AnalyticsOrganizationModel),
                [
                    {
                        "dataset_id": dataset.id,
                        "organization_id": organization_id,
                        "organization_name": organization_name,
                        "organization_url": organization_url,
                    }
                    for organization_id, organization_name, organization_url in verified.organizations
                ],
            )
            for batch in _feedback_batches(artifact_path, dataset.id, batch_size):
                await session.execute(insert(AnalyticsFeedbackModel), batch)
    except SQLAlchemyError:
        msg = "database rejected the analytical snapshot; transaction rolled back"
        raise AnalyticsLoadError(msg) from None
    return AnalyticsLoadResult(
        dataset_id=dataset.id,
        records=manifest.records,
        organizations=manifest.unique_organizations,
        already_loaded=False,
    )


async def _existing_result(
    session: AsyncSession,
    dataset: AnalyticsDatasetModel,
    manifest: FeedbackPreparationManifest,
) -> AnalyticsLoadResult:
    feedback_count = await session.scalar(
        select(func.count()).select_from(AnalyticsFeedbackModel).where(AnalyticsFeedbackModel.dataset_id == dataset.id),
    )
    organization_count = await session.scalar(
        select(func.count())
        .select_from(AnalyticsOrganizationModel)
        .where(AnalyticsOrganizationModel.dataset_id == dataset.id),
    )
    if feedback_count != manifest.records or organization_count != manifest.unique_organizations:
        msg = "existing dataset is incomplete or inconsistent with manifest"
        raise AnalyticsLoadError(msg)
    return AnalyticsLoadResult(
        dataset_id=dataset.id,
        records=manifest.records,
        organizations=manifest.unique_organizations,
        already_loaded=True,
    )
