import hashlib
import json
import os
from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.analytics import FeedbackArtifactError, FeedbackPreparationManifest, parse_feedback_line
from src.infrastructure.analytics_db import (
    ANALYTICS_SCHEMA,
    AnalyticsBase,
    AnalyticsDatasetModel,
    AnalyticsFeedbackModel,
)
from src.operations.feedback_loading import AnalyticsLoadError, load_feedback_snapshot, verify_feedback_artifact
from src.operations.feedback_preparation import prepare_yandex_feedback


def _row(review_id: str, *, organization_name: str = "Стоматология") -> dict[str, object]:
    return {
        "review_id": review_id,
        "review_text": "Отзыв пациента",
        "rating": 3,
        "author": "Приватный автор",
        "published_at": "2025-04-03",
        "org_id": "229248496819",
        "org_name": organization_name,
        "org_url": "https://yandex.ru/maps/org/229248496819/reviews/",
        "search_query": "стоматология",
    }


def _prepared_artifact(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source.jsonl"
    artifact = tmp_path / "feedback.jsonl"
    manifest = tmp_path / "feedback.manifest.json"
    source.write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False)
            for row in [_row("7995cc14c8b92196"), _row("6529392e51d48c89")]
        )
        + "\n",
        encoding="utf-8",
    )
    prepare_yandex_feedback(source, artifact, manifest)
    return artifact, manifest


def test_verify_feedback_artifact_reconciles_manifest(tmp_path: Path) -> None:
    artifact, manifest_path = _prepared_artifact(tmp_path)

    verified = verify_feedback_artifact(artifact, manifest_path)

    assert verified.manifest.records == 2
    assert verified.organizations == (
        (
            "229248496819",
            "Стоматология",
            "https://yandex.ru/maps/org/229248496819/reviews/",
        ),
    )


def test_verify_feedback_artifact_rejects_changed_bytes(tmp_path: Path) -> None:
    artifact, manifest = _prepared_artifact(tmp_path)
    artifact.write_bytes(artifact.read_bytes() + b"\n")

    with pytest.raises(AnalyticsLoadError, match="size does not match"):
        verify_feedback_artifact(artifact, manifest)


def test_canonical_reader_error_does_not_expose_feedback_text() -> None:
    private_text = "Секретные медицинские сведения"
    raw = json.dumps({"text": private_text}, ensure_ascii=False).encode()

    with pytest.raises(FeedbackArtifactError) as captured:
        parse_feedback_line(raw, line_number=9)

    assert "line 9" in str(captured.value)
    assert "content_hash=" in str(captured.value)
    assert private_text not in str(captured.value)


def test_verify_feedback_artifact_rejects_conflicting_organization_identity(tmp_path: Path) -> None:
    artifact, manifest_path = _prepared_artifact(tmp_path)
    records = [json.loads(line) for line in artifact.read_text(encoding="utf-8").splitlines()]
    records[1]["organization_name"] = "Другое название"
    artifact.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) for record in records) + "\n",
        encoding="utf-8",
    )
    manifest = FeedbackPreparationManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    manifest = manifest.model_copy(
        update={
            "output_size_bytes": artifact.stat().st_size,
            "output_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        },
    )
    manifest_path.write_text(manifest.model_dump_json(), encoding="utf-8")

    with pytest.raises(AnalyticsLoadError, match="identity is inconsistent"):
        verify_feedback_artifact(artifact, manifest_path)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_load_feedback_snapshot_is_transactional_and_idempotent(tmp_path: Path) -> None:
    database_url = os.getenv("TEST_DATABASE_URL")
    if database_url is None:
        pytest.skip("TEST_DATABASE_URL is not configured")
    artifact, manifest = _prepared_artifact(tmp_path)
    engine = create_async_engine(database_url)
    try:
        async with engine.begin() as connection:
            await connection.exec_driver_sql(f"DROP SCHEMA IF EXISTS {ANALYTICS_SCHEMA} CASCADE")
            await connection.exec_driver_sql(f"CREATE SCHEMA {ANALYTICS_SCHEMA}")
            await connection.run_sync(AnalyticsBase.metadata.create_all)
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        async with session_factory() as session:
            first = await load_feedback_snapshot(session, artifact, manifest, batch_size=1)
            second = await load_feedback_snapshot(session, artifact, manifest, batch_size=1)
            dataset_count = await session.scalar(select(func.count()).select_from(AnalyticsDatasetModel))
            feedback_count = await session.scalar(select(func.count()).select_from(AnalyticsFeedbackModel))

        assert first.already_loaded is False
        assert second.already_loaded is True
        assert second.dataset_id == first.dataset_id
        assert dataset_count == 1
        assert feedback_count == 2
    finally:
        async with engine.begin() as connection:
            await connection.exec_driver_sql(f"DROP SCHEMA IF EXISTS {ANALYTICS_SCHEMA} CASCADE")
        await engine.dispose()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_database_failure_rolls_back_without_exposing_feedback(tmp_path: Path) -> None:
    database_url = os.getenv("TEST_DATABASE_URL")
    if database_url is None:
        pytest.skip("TEST_DATABASE_URL is not configured")
    artifact, manifest = _prepared_artifact(tmp_path)
    private_text = "Отзыв пациента"
    engine = create_async_engine(database_url)
    try:
        async with engine.begin() as connection:
            await connection.exec_driver_sql(f"DROP SCHEMA IF EXISTS {ANALYTICS_SCHEMA} CASCADE")
            await connection.exec_driver_sql(f"CREATE SCHEMA {ANALYTICS_SCHEMA}")
            await connection.run_sync(AnalyticsBase.metadata.create_all)
            await connection.exec_driver_sql(
                "ALTER TABLE analytics.feedback "
                "ADD CONSTRAINT ck_test_reject_private_text "
                "CHECK (text <> 'Отзыв пациента')",
            )
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        async with session_factory() as session:
            with pytest.raises(AnalyticsLoadError) as captured:
                await load_feedback_snapshot(session, artifact, manifest, batch_size=1)
            dataset_count = await session.scalar(select(func.count()).select_from(AnalyticsDatasetModel))
            feedback_count = await session.scalar(select(func.count()).select_from(AnalyticsFeedbackModel))

        assert str(captured.value) == "database rejected the analytical snapshot; transaction rolled back"
        assert private_text not in str(captured.value)
        assert captured.value.__cause__ is None
        assert captured.value.__suppress_context__ is True
        assert dataset_count == 0
        assert feedback_count == 0
    finally:
        async with engine.begin() as connection:
            await connection.exec_driver_sql(f"DROP SCHEMA IF EXISTS {ANALYTICS_SCHEMA} CASCADE")
        await engine.dispose()
