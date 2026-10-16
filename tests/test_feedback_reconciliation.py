import json
import os
from pathlib import Path

import pytest
from sqlalchemy import update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.infrastructure.analytics_db import ANALYTICS_SCHEMA, AnalyticsBase, AnalyticsFeedbackModel
from src.operations.feedback_loading import load_feedback_snapshot
from src.operations.feedback_preparation import prepare_yandex_feedback
from src.operations.feedback_reconciliation import reconcile_feedback_snapshot


def _prepared_artifact(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source.jsonl"
    artifact = tmp_path / "feedback.jsonl"
    manifest = tmp_path / "feedback.manifest.json"
    rows = [
        {
            "review_id": "7995cc14c8b92196",
            "review_text": "Первый отзыв",
            "rating": 2,
            "author": "Автор",
            "published_at": "2025-04-03",
            "org_id": "229248496819",
            "org_name": "Стоматология",
            "org_url": "https://yandex.ru/maps/org/229248496819/reviews/",
            "search_query": "стоматология",
        },
        {
            "review_id": "6529392e51d48c89",
            "review_text": "Второй отзыв",
            "rating": 4,
            "author": "Другой автор",
            "published_at": "2025-04-04",
            "org_id": "229248496819",
            "org_name": "Стоматология",
            "org_url": "https://yandex.ru/maps/org/229248496819/reviews/",
            "search_query": "стоматология",
        },
    ]
    source.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    prepare_yandex_feedback(source, artifact, manifest)
    return artifact, manifest


@pytest.mark.integration
@pytest.mark.asyncio
async def test_reconciliation_detects_missing_valid_and_changed_snapshot(tmp_path: Path) -> None:
    database_url = os.getenv("TEST_DATABASE_URL")
    if database_url is None:
        pytest.skip("TEST_DATABASE_URL is not configured")
    artifact, manifest = _prepared_artifact(tmp_path)
    engine = create_async_engine(database_url)
    private_replacement = "Секретный изменённый текст"
    try:
        async with engine.begin() as connection:
            await connection.exec_driver_sql(f"DROP SCHEMA IF EXISTS {ANALYTICS_SCHEMA} CASCADE")
            await connection.exec_driver_sql(f"CREATE SCHEMA {ANALYTICS_SCHEMA}")
            await connection.run_sync(AnalyticsBase.metadata.create_all)
        session_factory = async_sessionmaker(engine, expire_on_commit=False)

        async with session_factory() as session:
            missing = await reconcile_feedback_snapshot(session, artifact, manifest)
        async with session_factory() as session:
            await load_feedback_snapshot(session, artifact, manifest)
        async with session_factory() as session:
            valid = await reconcile_feedback_snapshot(session, artifact, manifest)
        async with session_factory.begin() as session:
            await session.execute(
                update(AnalyticsFeedbackModel)
                .where(AnalyticsFeedbackModel.feedback_id == "yandex_maps:7995cc14c8b92196")
                .values(text=private_replacement),
            )
        async with session_factory() as session:
            changed = await reconcile_feedback_snapshot(session, artifact, manifest)

        assert missing.errors == ("dataset.not_found",)
        assert valid.is_valid
        assert valid.matching_records == 2
        assert changed.is_valid is False
        assert changed.changed_records == 1
        assert changed.errors == ("feedback.records_changed",)
        assert private_replacement not in changed.model_dump_json()
    finally:
        async with engine.begin() as connection:
            await connection.exec_driver_sql(f"DROP SCHEMA IF EXISTS {ANALYTICS_SCHEMA} CASCADE")
        await engine.dispose()
