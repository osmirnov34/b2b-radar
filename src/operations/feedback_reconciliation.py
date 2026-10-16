"""Privacy-safe reconciliation of canonical artifacts and PostgreSQL snapshots."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from src.analytics import (
    FeedbackPreparationManifest,
    FeedbackReconciliationReport,
    FeedbackRecord,
    FeedbackSource,
    iter_feedback_records,
)
from src.infrastructure.analytics_db import (
    AnalyticsDatasetModel,
    AnalyticsFeedbackModel,
    AnalyticsOrganizationModel,
)
from src.operations.feedback_loading import AnalyticsLoadError, verify_feedback_artifact

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import date
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncSession


def _fingerprint(record: FeedbackRecord) -> str:
    return hashlib.sha256(record.model_dump_json().encode()).hexdigest()


def _artifact_fingerprints(path: Path) -> dict[str, str]:
    return {record.feedback_id: _fingerprint(record) for record in iter_feedback_records(path)}


@dataclass(frozen=True)
class _StoredFeedback:
    feedback_id: str
    source_record_id: str
    text: str
    rating: int
    published_on: date
    organization_id: str
    search_query: str
    organization_name: str
    organization_url: str


@dataclass(frozen=True)
class _DatabaseSummary:
    records: int
    organizations: int
    rating_distribution: dict[int, int]
    published_on_min: date | None
    published_on_max: date | None
    missing_records: int
    unexpected_records: int
    changed_records: int


async def reconcile_feedback_snapshot(
    session: AsyncSession,
    artifact_path: Path,
    manifest_path: Path,
) -> FeedbackReconciliationReport:
    """Compare all canonical fields with the stored snapshot using only aggregate diagnostics."""
    verified = verify_feedback_artifact(artifact_path, manifest_path)
    manifest = verified.manifest
    artifact_fingerprints = _artifact_fingerprints(artifact_path)
    try:
        dataset = await session.scalar(
            select(AnalyticsDatasetModel).where(AnalyticsDatasetModel.artifact_sha256 == manifest.output_sha256),
        )
        if dataset is None:
            return _missing_dataset_report(manifest)
        raw_rows = (
            await session.execute(
                select(
                    AnalyticsFeedbackModel.feedback_id,
                    AnalyticsFeedbackModel.source_record_id,
                    AnalyticsFeedbackModel.text,
                    AnalyticsFeedbackModel.rating,
                    AnalyticsFeedbackModel.published_on,
                    AnalyticsFeedbackModel.organization_id,
                    AnalyticsFeedbackModel.search_query,
                    AnalyticsOrganizationModel.organization_name,
                    AnalyticsOrganizationModel.organization_url,
                ).join(
                    AnalyticsOrganizationModel,
                    (AnalyticsOrganizationModel.dataset_id == AnalyticsFeedbackModel.dataset_id)
                    & (AnalyticsOrganizationModel.organization_id == AnalyticsFeedbackModel.organization_id),
                ).where(AnalyticsFeedbackModel.dataset_id == dataset.id),
            )
        ).all()
        rows = [
            _StoredFeedback(
                feedback_id=row.feedback_id,
                source_record_id=row.source_record_id,
                text=row.text,
                rating=row.rating,
                published_on=row.published_on,
                organization_id=row.organization_id,
                search_query=row.search_query,
                organization_name=row.organization_name,
                organization_url=row.organization_url,
            )
            for row in raw_rows
        ]
        organizations = (
            await session.scalars(
                select(AnalyticsOrganizationModel.organization_id).where(
                    AnalyticsOrganizationModel.dataset_id == dataset.id,
                ),
            )
        ).all()
        return _build_report(manifest, dataset, rows, organizations, artifact_fingerprints)
    except SQLAlchemyError:
        msg = "database reconciliation failed without exposing snapshot contents"
        raise AnalyticsLoadError(msg) from None


def _build_report(
    manifest: FeedbackPreparationManifest,
    dataset: AnalyticsDatasetModel,
    rows: Sequence[_StoredFeedback],
    organizations: Sequence[str],
    artifact_fingerprints: dict[str, str],
) -> FeedbackReconciliationReport:
    try:
        source = FeedbackSource(dataset.source)
    except ValueError as exc:
        msg = "stored dataset source is unsupported"
        raise AnalyticsLoadError(msg) from exc

    database_fingerprints: dict[str, str] = {}
    ratings: Counter[int] = Counter()
    published_dates: list[date] = []
    for row in rows:
        record = FeedbackRecord(
            feedback_id=row.feedback_id,
            source=source,
            source_record_id=row.source_record_id,
            text=row.text,
            rating=row.rating,
            published_on=row.published_on,
            organization_id=row.organization_id,
            organization_name=row.organization_name,
            organization_url=row.organization_url,
            search_query=row.search_query,
        )
        database_fingerprints[record.feedback_id] = _fingerprint(record)
        ratings[record.rating] += 1
        published_dates.append(record.published_on)

    artifact_ids = set(artifact_fingerprints)
    database_ids = set(database_fingerprints)
    common_ids = artifact_ids & database_ids
    changed = sum(artifact_fingerprints[item] != database_fingerprints[item] for item in common_ids)
    summary = _DatabaseSummary(
        records=len(rows),
        organizations=len(organizations),
        rating_distribution=dict(sorted(ratings.items())),
        published_on_min=min(published_dates, default=None),
        published_on_max=max(published_dates, default=None),
        missing_records=len(artifact_ids - database_ids),
        unexpected_records=len(database_ids - artifact_ids),
        changed_records=changed,
    )
    errors = _reconciliation_errors(manifest, dataset, summary)
    return FeedbackReconciliationReport(
        validated_at=datetime.now(UTC),
        artifact_sha256=manifest.output_sha256,
        dataset_id=dataset.id,
        expected_records=manifest.records,
        database_records=summary.records,
        matching_records=len(common_ids) - changed,
        missing_records=summary.missing_records,
        unexpected_records=summary.unexpected_records,
        changed_records=summary.changed_records,
        expected_organizations=manifest.unique_organizations,
        database_organizations=summary.organizations,
        expected_rating_distribution=manifest.rating_distribution,
        database_rating_distribution=summary.rating_distribution,
        expected_published_on_min=manifest.published_on_min,
        expected_published_on_max=manifest.published_on_max,
        database_published_on_min=summary.published_on_min,
        database_published_on_max=summary.published_on_max,
        errors=tuple(errors),
    )


def _reconciliation_errors(
    manifest: FeedbackPreparationManifest,
    dataset: AnalyticsDatasetModel,
    summary: _DatabaseSummary,
) -> list[str]:
    errors: list[str] = []
    metadata_matches = (
        dataset.source == manifest.source.value
        and dataset.manifest_schema_version == manifest.schema_version
        and dataset.source_sha256 == manifest.source_sha256
        and dataset.artifact_sha256 == manifest.output_sha256
        and dataset.prepared_at == manifest.prepared_at
        and dataset.record_count == manifest.records
        and dataset.organization_count == manifest.unique_organizations
        and dataset.published_on_min == manifest.published_on_min
        and dataset.published_on_max == manifest.published_on_max
    )
    if not metadata_matches:
        errors.append("dataset.metadata_mismatch")
    if summary.records != manifest.records:
        errors.append("feedback.count_mismatch")
    if summary.organizations != manifest.unique_organizations:
        errors.append("organizations.count_mismatch")
    if summary.rating_distribution != manifest.rating_distribution:
        errors.append("feedback.rating_distribution_mismatch")
    if (summary.published_on_min, summary.published_on_max) != (
        manifest.published_on_min,
        manifest.published_on_max,
    ):
        errors.append("feedback.publication_range_mismatch")
    if summary.missing_records:
        errors.append("feedback.records_missing")
    if summary.unexpected_records:
        errors.append("feedback.records_unexpected")
    if summary.changed_records:
        errors.append("feedback.records_changed")
    return errors


def _missing_dataset_report(manifest: FeedbackPreparationManifest) -> FeedbackReconciliationReport:
    return FeedbackReconciliationReport(
        validated_at=datetime.now(UTC),
        artifact_sha256=manifest.output_sha256,
        expected_records=manifest.records,
        database_records=0,
        matching_records=0,
        missing_records=manifest.records,
        unexpected_records=0,
        changed_records=0,
        expected_organizations=manifest.unique_organizations,
        database_organizations=0,
        expected_rating_distribution=manifest.rating_distribution,
        database_rating_distribution={},
        expected_published_on_min=manifest.published_on_min,
        expected_published_on_max=manifest.published_on_max,
        errors=("dataset.not_found",),
    )
