import importlib.util
import os
from pathlib import Path
from types import ModuleType

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.infrastructure.analytics_db import AnalyticsBase
from src.operations.problem_priorities import load_problem_priorities


def _migration(filename: str) -> ModuleType:
    path = Path(__file__).parents[1] / "migrations/versions" / filename
    spec = importlib.util.spec_from_file_location(filename, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.integration
@pytest.mark.asyncio
async def test_reviewed_kpis_exclude_rejected_and_preserve_denominator() -> None:  # noqa: PLR0915
    url = os.getenv("TEST_DATABASE_URL")
    if url is None:
        pytest.skip("TEST_DATABASE_URL is not configured")
    engine = create_async_engine(url)
    migration = _migration("d0e1f2a3b4c5_fix_reviewed_problem_kpis.py")
    try:
        async with engine.begin() as connection:
            await connection.exec_driver_sql("DROP SCHEMA IF EXISTS analytics CASCADE")
            await connection.exec_driver_sql("CREATE SCHEMA analytics")
            await connection.run_sync(AnalyticsBase.metadata.create_all)
            await connection.exec_driver_sql("""
                INSERT INTO analytics.datasets
                (id, source, manifest_schema_version, source_sha256, artifact_sha256, prepared_at,
                 record_count, organization_count)
                VALUES (1, 'yandex_maps', 1, repeat('a', 64), repeat('b', 64), now(), 3, 1)
            """)
            await connection.exec_driver_sql("""
                INSERT INTO analytics.organizations VALUES (1, 'clinic', 'Клиника', 'https://example.org')
            """)
            await connection.exec_driver_sql("""
                INSERT INTO analytics.feedback
                (dataset_id, feedback_id, source_record_id, organization_id, text, rating, published_on)
                SELECT 1, 'review-' || n, 'source-' || n, 'clinic', 'Private text', 1, '2026-09-01'::date
                FROM generate_series(1, 3) AS n
            """)
            await connection.exec_driver_sql("""
                INSERT INTO analytics.problem_runs
                (id, dataset_id, artifact_sha256, taxonomy_id, taxonomy_version, ruleset_id, ruleset_version,
                 generated_at, feedback_count, assessed_feedback_count, assessment_count)
                VALUES (1, 1, repeat('c', 64), 'dentistry', '1.0.0', 'rules', '1.0.0',
                        '2026-10-16T00:00:00Z', 3, 3, 3)
            """)
            await connection.exec_driver_sql("""
                INSERT INTO analytics.problem_categories
                VALUES (1, 'waiting_time', 'Ожидание', 'Ожидание приёма', '[]', true)
            """)
            await connection.exec_driver_sql("""
                INSERT INTO analytics.problem_assessments
                (run_id, dataset_id, feedback_id, category_id, stance, category_probability,
                 stance_probability, method, evidence_spans, review_status, assessed_at)
                VALUES
                (1, 1, 'review-1', 'waiting_time', 'complaint', 0.8, 0.9, 'rule', '[]', 'accepted', now()),
                (1, 1, 'review-2', 'waiting_time', 'complaint', 0.6, 0.9, 'rule', '[]', 'unreviewed', now()),
                (1, 1, 'review-3', 'waiting_time', 'complaint', 1.0, 0.9, 'rule', '[]', 'rejected', now())
            """)
            # Execute the old contracts first: CREATE OR REPLACE must preserve their column order and types.
            for monthly in (False, True):
                await connection.exec_driver_sql(migration._view_sql(monthly=monthly, corrected=False))
                await connection.exec_driver_sql(migration._view_sql(monthly=monthly, corrected=True))
            result = await connection.exec_driver_sql("SELECT * FROM analytics.problem_kpis")
            row = result.mappings().one()
            assert row["feedback_count"] == 3
            assert row["complaint_count"] == 2
            assert row["complaint_share"] == pytest.approx(2 / 3)
            assert row["mean_category_probability"] == pytest.approx(0.7)
            assert row["rejected_count"] == 1
            assert row["accepted_complaint_count"] == 1
            assert row["unreviewed_complaint_count"] == 1
            monthly_result = await connection.exec_driver_sql("SELECT * FROM analytics.problem_monthly_kpis")
            assert monthly_result.mappings().one()["complaint_share"] == pytest.approx(2 / 3)
        async with async_sessionmaker(engine)() as session:
            priorities = await load_problem_priorities(session, 1)
            assert priorities[0].signal.complaint_count == 2
            assert set(priorities[0].signal.evidence_feedback_ids) == {"review-1", "review-2"}
            assert priorities[0].status == "review_required"
            assert not priorities[0].trend_available
        async with engine.begin() as connection:
            await connection.exec_driver_sql("""
                INSERT INTO analytics.feedback
                (dataset_id, feedback_id, source_record_id, organization_id, text, rating, published_on)
                SELECT 1, 'extra-' || n, 'extra-source-' || n, 'clinic', 'Private text', 4,
                       CASE WHEN n <= 30 THEN '2026-08-01'::date ELSE '2026-09-01'::date END
                FROM generate_series(1, 60) AS n
            """)
        async with async_sessionmaker(engine)() as session:
            trend_priorities = await load_problem_priorities(session, 1)
            signal = trend_priorities[0].signal
            assert signal.previous_feedback_count == 30
            assert signal.current_feedback_count == 33
            assert signal.previous_complaint_share == 0
            assert signal.current_complaint_share == pytest.approx(2 / 33)
            assert trend_priorities[0].trend_available
        async with engine.begin() as connection:
            await connection.exec_driver_sql("UPDATE analytics.problem_assessments SET review_status = 'rejected'")
        async with async_sessionmaker(engine)() as session:
            assert await load_problem_priorities(session, 1) == []
        async with engine.begin() as connection:
            # Downgrade removes appended columns; upgrading again restores the corrected contract.
            for monthly in (True, False):
                view = "problem_monthly_kpis" if monthly else "problem_kpis"
                await connection.exec_driver_sql(f"DROP VIEW analytics.{view}")
                await connection.exec_driver_sql(migration._view_sql(monthly=monthly, corrected=False))
                old = await connection.exec_driver_sql(
                    f"SELECT complaint_count FROM analytics.{view}",  # noqa: S608 -- view is a fixed test constant.
                )
                assert old.scalar_one() == 3
                await connection.exec_driver_sql(migration._view_sql(monthly=monthly, corrected=True))
                corrected = await connection.exec_driver_sql(
                    f"SELECT complaint_count FROM analytics.{view}",  # noqa: S608 -- view is a fixed test constant.
                )
                assert corrected.scalar_one() == 0
    finally:
        async with engine.begin() as connection:
            await connection.exec_driver_sql("DROP SCHEMA IF EXISTS analytics CASCADE")
        await engine.dispose()
