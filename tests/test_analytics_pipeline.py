import hashlib
import importlib.util
import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.infrastructure.analytics_db import AnalyticsBase
from src.operations import analytics_pipeline as pipeline
from tests.test_problem_loading import _artifacts
from tests.test_reviewed_kpis import _migration

PROJECT_ROOT = Path(__file__).parents[1]


def test_pipeline_config_rejects_partial_ml_input_and_path_traversal() -> None:
    with pytest.raises(ValidationError, match="configured together"):
        pipeline.AnalyticsPipelineConfig(source=Path("source"), run_name="run", ml_corpus_manifest=Path("corpus"))
    with pytest.raises(ValidationError, match="run_name"):
        pipeline.AnalyticsPipelineConfig(source=Path("source"), run_name="../outside")


@pytest.mark.integration
async def test_pipeline_full_run_resume_and_failure_recovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: PLR0915
    url = os.getenv("TEST_DATABASE_URL")
    if url is None:
        pytest.skip("TEST_DATABASE_URL is not configured")
    _artifacts(tmp_path)
    config = pipeline.AnalyticsPipelineConfig(
        source=tmp_path / "reviews.jsonl",
        run_name="test-001",
        runs_root=tmp_path / "runs",
        create_pdf=importlib.util.find_spec("reportlab") is not None,
    )
    if config.create_pdf and not config.font.is_file():
        config = config.model_copy(update={"create_pdf": False})
    root = config.runs_root / config.run_name
    engine = create_async_engine(url)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with engine.begin() as connection:
            await connection.exec_driver_sql("DROP SCHEMA IF EXISTS analytics_bi CASCADE")
            await connection.exec_driver_sql("DROP SCHEMA IF EXISTS analytics CASCADE")
            await connection.exec_driver_sql("CREATE SCHEMA analytics")
            await connection.run_sync(AnalyticsBase.metadata.create_all)
            for revision in (
                "c9d0e1f2a3b4_add_analytics_kpi_marts.py",
                "d0e1f2a3b4c5_fix_reviewed_problem_kpis.py",
                "e1f2a3b4c5d6_add_datalens_sources.py",
            ):
                migration = _migration(revision)
                migration.op = MagicMock()
                migration.upgrade()
                for call in migration.op.execute.call_args_list:
                    await connection.exec_driver_sql(call.args[0])
        preflight = await pipeline.preflight_analytics_pipeline(config, factory)
        assert preflight["mode"] == "rules_only"
        assert not root.exists()
        partial = await pipeline.run_analytics_pipeline(config, factory, stop_after=pipeline.AnalyticsStage.PREPARE)
        assert partial["status"] == "partial"
        original = config.source.read_bytes()
        before = (root / "pipeline-state.json").read_bytes()
        config.source.write_bytes(original + b"\n")
        with pytest.raises(pipeline.AnalyticsPipelineError, match="inputs or configuration changed"):
            await pipeline.run_analytics_pipeline(config, factory, resume=True)
        assert (root / "pipeline-state.json").read_bytes() == before
        config.source.write_bytes(original)
        original_stage = pipeline._execute_stage

        async def fail_after_assess(*args: object) -> None:
            await original_stage(*args)
            if args[0] == pipeline.AnalyticsStage.ASSESS:
                msg = "simulated interruption after publication"
                raise RuntimeError(msg)

        monkeypatch.setattr(pipeline, "_execute_stage", fail_after_assess)
        with pytest.raises(RuntimeError, match="simulated interruption"):
            await pipeline.run_analytics_pipeline(config, factory, resume=True)
        assert json.loads((root / "pipeline-state.json").read_text())["status"] == "failed"
        assert not (root / ".pipeline.lock").exists()
        monkeypatch.setattr(pipeline, "_execute_stage", original_stage)
        result = await pipeline.run_analytics_pipeline(config, factory, resume=True)
        assert result["status"] == "completed"
        assert result["dataset_id"] == 1
        assert result["problem_run_id"] == 1
        assert list((root / "attempts").glob("05-assess-*"))
        package = root / "07-datalens/package"
        assert "WHERE run_id = 1" in (package / "category_overview.sql").read_text()
        pdf = root / "08-pdf/report/report.pdf"
        pdf_hash = hashlib.sha256(pdf.read_bytes()).hexdigest() if config.create_pdf else None
        completed = await pipeline.run_analytics_pipeline(config, factory, resume=True)
        assert completed["status"] == "completed"
        if config.create_pdf:
            assert hashlib.sha256(pdf.read_bytes()).hexdigest() == pdf_hash
        with pytest.raises(FileExistsError):
            await pipeline.run_analytics_pipeline(config, factory)
        (root / ".pipeline.lock").touch()
        with pytest.raises(pipeline.AnalyticsPipelineError, match="locked"):
            await pipeline.run_analytics_pipeline(config, factory, resume=True)
        (root / ".pipeline.lock").unlink()
        canonical = root / "01-prepare/feedback.jsonl"
        canonical_bytes = canonical.read_bytes()
        canonical.write_bytes(canonical_bytes + b"\n")
        with pytest.raises(pipeline.AnalyticsPipelineError, match="artifacts changed"):
            await pipeline.run_analytics_pipeline(config, factory, resume=True)
        canonical.write_bytes(canonical_bytes)
        async with engine.begin() as connection:
            await connection.exec_driver_sql("UPDATE analytics.problem_assessments SET review_status = 'rejected'")
        with pytest.raises(pipeline.AnalyticsPipelineError, match="assessments changed"):
            await pipeline.run_analytics_pipeline(config, factory, resume=True)
    finally:
        async with engine.begin() as connection:
            await connection.exec_driver_sql("DROP SCHEMA IF EXISTS analytics_bi CASCADE")
            await connection.exec_driver_sql("DROP SCHEMA IF EXISTS analytics CASCADE")
        await engine.dispose()
