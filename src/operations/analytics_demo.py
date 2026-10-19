"""Initialize and inspect an isolated local analytics demonstration."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
from datetime import UTC, datetime
from http import HTTPStatus
from typing import TYPE_CHECKING

from dotenv import dotenv_values, load_dotenv
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from src.analytics.report import AnalyticalReport
from src.config import get_analytics_api_settings, get_settings
from src.operations.analytics_pipeline import AnalyticsPipelineConfig

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


def initialize_analytics_demo(source: Path, directory: Path) -> None:
    """Create private local credentials and configuration without replacing user files."""
    if not source.is_file():
        msg = "demo source file is missing"
        raise FileNotFoundError(msg)
    directory.mkdir(parents=True, mode=0o700, exist_ok=False)
    environment = {
        "POSTGRES_USER": "analytics_demo",
        "POSTGRES_DB": "analytics_demo",
        "POSTGRES_PASSWORD": secrets.token_urlsafe(32),
        "POSTGRES_HOST": "127.0.0.1",
        "POSTGRES_PORT": "55440",
        "ANALYTICS_API_TOKEN": secrets.token_urlsafe(32),
        "ANALYTICS_PRIVATE_TOKEN": secrets.token_urlsafe(32),
    }
    target = directory / "environment.env"
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write("".join(f"{key}={value}\n" for key, value in environment.items()))
    config = AnalyticsPipelineConfig(
        source=source.resolve(), runs_root=(directory / "runs").resolve(), run_name="acceptance-001"
    )
    (directory / "pipeline.json").write_text(config.model_dump_json(indent=2) + "\n", encoding="utf-8")


def load_demo_environment(directory: Path) -> AnalyticsPipelineConfig:
    """Restrict demo operations to an explicit loopback database configuration."""
    path = directory / "environment.env"
    values = dotenv_values(path)
    if (
        values.get("POSTGRES_HOST") != "127.0.0.1"
        or values.get("POSTGRES_DB") != "analytics_demo"
        or values.get("POSTGRES_USER") != "analytics_demo"
    ):
        msg = "demo commands require the isolated local analytics_demo database"
        raise ValueError(msg)
    load_dotenv(path, override=True)
    get_settings.cache_clear()
    get_analytics_api_settings.cache_clear()
    return AnalyticsPipelineConfig.model_validate_json((directory / "pipeline.json").read_text(encoding="utf-8"))


def _verify_artifacts(root: Path, state: dict[str, Any]) -> None:
    from src.operations.analytics_pipeline import AnalyticsStage  # noqa: PLC0415 -- runtime stage enumeration.

    for index, stage in enumerate(AnalyticsStage, start=1):
        directory = root / f"{index:02d}-{stage.value}"
        actual = {
            str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in directory.rglob("*")
            if path.is_file()
        }
        if actual != state["completed"].get(stage.value):
            msg = "demo artifacts do not match completed checkpoints"
            raise ValueError(msg)


async def _inspect_api(dataset_id: int, run_id: int) -> bool:
    """Check access boundaries and aggregate routes without starting a public listener."""
    from httpx import ASGITransport, AsyncClient  # noqa: PLC0415 -- acceptance-only dependency.

    from src.web.analytics_app import create_analytics_app  # noqa: PLC0415 -- delivery boundary for acceptance.

    credentials = get_analytics_api_settings()
    if credentials.analytics_api_token is None or credentials.analytics_private_token is None:
        msg = "demo API credentials are missing"
        raise ValueError(msg)
    async with AsyncClient(transport=ASGITransport(app=create_analytics_app()), base_url="http://local-demo") as client:
        unauthorized = await client.get(f"/api/analytics/summary?dataset_id={dataset_id}")
        if unauthorized.status_code != HTTPStatus.UNAUTHORIZED:
            msg = "API unexpectedly allows anonymous access"
            raise ValueError(msg)
        headers = {"Authorization": "Bearer " + credentials.analytics_api_token.get_secret_value()}
        for path in (
            f"summary?dataset_id={dataset_id}",
            f"organizations?dataset_id={dataset_id}&limit=1",
            f"problems?run_id={run_id}&limit=1",
            f"monthly?dataset_id={dataset_id}&limit=1",
            f"problem-monthly?run_id={run_id}&limit=1",
            f"priorities?run_id={run_id}&limit=1",
        ):
            response = await client.get("/api/analytics/" + path, headers=headers)
            if response.status_code != HTTPStatus.OK:
                msg = "aggregate API request failed local acceptance"
                raise ValueError(msg)
        private = await client.get(
            f"/api/analytics/private/examples?run_id={run_id}&category_id=waiting_time", headers=headers
        )
        if private.status_code != HTTPStatus.UNAUTHORIZED:
            msg = "aggregate token unexpectedly permits private examples"
            raise ValueError(msg)
    return True


async def inspect_analytics_demo(
    config: AnalyticsPipelineConfig,
    factory: async_sessionmaker[AsyncSession],
) -> dict[str, object]:
    """Produce an aggregate local-acceptance result; cloud and DataLens remain pending."""
    root = config.runs_root / config.run_name
    state = json.loads((root / "pipeline-state.json").read_text())
    if state.get("status") != "completed" or not config.create_pdf:
        msg = "demo acceptance requires a completed run with PDF output"
        raise ValueError(msg)
    _verify_artifacts(root, state)
    report = AnalyticalReport.model_validate_json((root / "08-pdf/report/report.json").read_text())
    async with factory() as session, session.begin():
        await session.execute(text("SET TRANSACTION READ ONLY"))
        migration = await session.scalar(text("SELECT version_num FROM alembic_version"))
        if migration != "e1f2a3b4c5d6":
            msg = "unexpected demo database migration revision"
            raise ValueError(msg)
        count = await session.scalar(
            text("SELECT count(*) FROM analytics.feedback WHERE dataset_id = :id"), {"id": state["dataset_id"]}
        )
        if count != report.feedback_count or report.run_id != state["problem_run_id"]:
            msg = "demo report and database identities or counts do not match"
            raise ValueError(msg)
        await session.execute(text("SET LOCAL ROLE b2b_radar_bi_reader"))
        bi_count = await session.scalar(
            text("SELECT sum(feedback_count) FROM analytics_bi.organization_overview WHERE run_id = :id"),
            {"id": report.run_id},
        )
        if bi_count != report.feedback_count:
            msg = "BI reader does not see the expected aggregate data"
            raise ValueError(msg)
        try:
            async with session.begin_nested():
                await session.execute(text("SELECT text FROM analytics.feedback LIMIT 1"))
        except DBAPIError as exc:
            if getattr(exc.orig, "sqlstate", None) != "42501":
                raise
            private_blocked = True
        else:
            private_blocked = False
        if not private_blocked:
            msg = "BI reader unexpectedly has access to private reviews"
            raise ValueError(msg)
    api_verified = await _inspect_api(report.dataset_id, report.run_id)
    return {
        "schema_version": 1,
        "checked_at": datetime.now(UTC).isoformat(),
        "status": "local_acceptance_only",
        "dataset_id": report.dataset_id,
        "problem_run_id": report.run_id,
        "feedback_count": report.feedback_count,
        "organizations": len(report.organizations),
        "categories": len(report.categories),
        "source_sha256": state["signature"]["inputs"]["source"],
        "dataset_sha256": report.dataset_sha256,
        "assessments_sha256": report.assessments_sha256,
        "pdf_sha256": hashlib.sha256((root / "08-pdf/report/report.pdf").read_bytes()).hexdigest(),
        "checkpoints_verified": True,
        "api_access_verified": api_verified,
        "bi_private_access_blocked": private_blocked,
        "cloud_deployment": "pending",
        "datalens_dashboard": "pending_manual_assembly",
        "assessment_mode": "hybrid" if config.ml_corpus_manifest else "rules_only",
    }
