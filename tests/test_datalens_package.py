import hashlib
import importlib.util
import json
import re
from pathlib import Path

import pytest
from pydantic import ValidationError
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from src.operations.datalens_package import DashboardSpec, build_datalens_package, dataset_sql

PROJECT_ROOT = Path(__file__).parents[1]
SPEC_PATH = PROJECT_ROOT / "configs/datalens-dashboard.v1.json"


def test_package_is_reproducible_and_refuses_overwrite(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    manifest = build_datalens_package(first, 7)
    assert build_datalens_package(second, 7) == manifest
    for path in first.iterdir():
        assert path.read_bytes() == (second / path.name).read_bytes()
    persisted = json.loads((first / "manifest.json").read_text())
    for filename, checksum in persisted["files_sha256"].items():
        assert hashlib.sha256((first / filename).read_bytes()).hexdigest() == checksum
    assert "WHERE run_id = 7" in (first / "category_overview.sql").read_text()
    with pytest.raises(FileExistsError):
        build_datalens_package(first, 8)
    assert "WHERE run_id = 7" in (first / "category_overview.sql").read_text()


@pytest.mark.parametrize("run_id", [0, -1, True, "1 OR 1=1"])
def test_query_requires_positive_integer_run(run_id: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        dataset_sql("category_overview", run_id)  # type: ignore[arg-type]


def test_spec_rejects_category_ratio_without_category_dimension() -> None:
    data = json.loads(SPEC_PATH.read_text())
    chart = next(chart for chart in data["charts"] if chart["id"] == "complaints")
    chart["dimensions"] = []
    with pytest.raises(ValidationError, match="category_id"):
        DashboardSpec.model_validate(data)


async def assert_datalens_sources(engine: AsyncEngine) -> None:
    """Check real SQL, weighted denominators and BI-role isolation on the shared regression fixture."""
    spec = DashboardSpec.model_validate_json(SPEC_PATH.read_text())
    path = PROJECT_ROOT / "migrations/versions/e1f2a3b4c5d6_add_datalens_sources.py"
    module_spec = importlib.util.spec_from_file_location("datalens_sources", path)
    assert module_spec is not None
    assert module_spec.loader is not None
    migration = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(migration)
    async with engine.connect() as connection, connection.begin():
        nested = await connection.begin_nested()
        await connection.exec_driver_sql("CREATE SCHEMA analytics_bi")
        await connection.exec_driver_sql("""
            INSERT INTO analytics.organizations VALUES (1, 'clinic-no-signals', 'Other clinic', 'https://example.org/2')
        """)
        await connection.exec_driver_sql("""
            INSERT INTO analytics.feedback
            (dataset_id, feedback_id, source_record_id, organization_id, text, rating, published_on)
            SELECT 1, 'no-signal-' || n, 'no-signal-source-' || n, 'clinic-no-signals',
                   'Private text', 5, '2026-09-01'::date FROM generate_series(1, 7) n
        """)
        await connection.exec_driver_sql("""
            INSERT INTO analytics.problem_categories VALUES (1, 'pricing', 'Цена', 'Цена', '[]', true)
        """)
        await connection.exec_driver_sql("""
            INSERT INTO analytics.problem_runs
            (id, dataset_id, artifact_sha256, taxonomy_id, taxonomy_version, ruleset_id, ruleset_version,
             generated_at, feedback_count, assessed_feedback_count, assessment_count)
            VALUES (2, 1, repeat('d', 64), 'dentistry', '1.0.0', 'rules', '1.0.0', now(), 10, 0, 0)
        """)
        columns: dict[str, set[str]] = {}
        for monthly in (False, True):
            for categories in (False, True):
                await connection.exec_driver_sql(migration.source_sql(monthly=monthly, categories=categories))
        for source in spec.datasets:
            result = await connection.exec_driver_sql(dataset_sql(source, 1))
            columns[source] = set(result.keys())
            assert not columns[source] & {"text", "feedback_id", "evidence_spans", "author", "reviewer_id"}
            rows = result.mappings().all()
            assert {row["run_id"] for row in rows} == {1}
            assert len(rows) == (4 if source.startswith("category") else 2)
            if source.startswith("category"):
                waiting = [row for row in rows if row["category_id"] == "waiting_time"]
                assert sum(row["feedback_count"] for row in waiting) == 10
                assert sum(row["complaint_count"] for row in waiting) / 10 == pytest.approx(0.2)
                assert sum(row["accepted_complaint_count"] for row in waiting) == 1
                assert all(row["complaint_count"] == 0 for row in rows if row["category_id"] == "pricing")
            else:
                assert sum(row["rating_sum"] for row in rows) / 10 == pytest.approx(3.8)
        for chart in spec.charts:
            required = set(chart.dimensions)
            for metric in chart.metrics:
                required.update(re.findall(r"\[([^]]+)\]", spec.metrics[metric]))
            assert required <= columns[chart.dataset]
        await _assert_bi_permissions(connection, spec)
        for name in ("category_monthly", "organization_monthly", "category_overview", "organization_overview"):
            await connection.exec_driver_sql(f"DROP VIEW analytics_bi.{name}")
        await connection.exec_driver_sql("DROP SCHEMA analytics_bi")
        await nested.rollback()


async def _assert_bi_permissions(connection: AsyncConnection, spec: DashboardSpec) -> None:
    role_sql = (PROJECT_ROOT / "scripts/datalens_reader_role.sql").read_text()
    statements = "\n".join(line for line in role_sql.splitlines() if not line.lstrip().startswith("--"))
    for statement in statements.split(";"):
        if statement.strip():
            await connection.exec_driver_sql(statement)
    await connection.exec_driver_sql("SET LOCAL ROLE b2b_radar_bi_reader")
    for source in spec.datasets:
        await connection.exec_driver_sql(dataset_sql(source, 1))
    with pytest.raises(DBAPIError):
        async with connection.begin_nested():
            await connection.exec_driver_sql("SELECT text FROM analytics.feedback")
    with pytest.raises(DBAPIError):
        async with connection.begin_nested():
            await connection.exec_driver_sql("DELETE FROM analytics_bi.category_overview")
    await connection.exec_driver_sql("RESET ROLE")
