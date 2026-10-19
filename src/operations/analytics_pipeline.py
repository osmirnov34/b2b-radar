"""Checkpointed orchestration of the independent analytical reporting workflow."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from enum import StrEnum
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import text

from src.analytics.pdf_report import write_analytical_report
from src.operations.analytical_report import load_analytical_report
from src.operations.datalens_package import build_datalens_package
from src.operations.feedback_loading import load_feedback_snapshot
from src.operations.feedback_preparation import prepare_yandex_feedback
from src.operations.feedback_reconciliation import reconcile_feedback_snapshot
from src.operations.problem_assessment import assess_feedback_problems
from src.operations.problem_loading import load_problem_assessments

if TYPE_CHECKING:
    from collections.abc import Callable

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


class AnalyticsStage(StrEnum):
    PREPARE = "prepare"
    LOAD_FEEDBACK = "load_feedback"
    RECONCILE = "reconcile"
    ADAPT_ML = "adapt_ml"
    ASSESS = "assess"
    LOAD_PROBLEMS = "load_problems"
    DATALENS = "datalens"
    PDF = "pdf"


class AnalyticsPipelineConfig(BaseModel):
    """Local paths only; database credentials remain in environment settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: Path
    runs_root: Path = Path("data/analytics-runs")
    run_name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
    taxonomy: Path = Path("configs/dental-problem-taxonomy.v1.json")
    rules: Path = Path("configs/dental-problem-rules.v1.json")
    font: Path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    create_pdf: bool = True
    ml_corpus_manifest: Path | None = None
    ml_reassignment_manifest: Path | None = None

    @model_validator(mode="after")
    def validate_ml_pair(self) -> AnalyticsPipelineConfig:
        if (self.ml_corpus_manifest is None) != (self.ml_reassignment_manifest is None):
            msg = "ML corpus and reassignment manifests must be configured together"
            raise ValueError(msg)
        return self


class AnalyticsPipelineError(RuntimeError):
    """Safe orchestration failure that does not expose source text or database secrets."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_hashes(config: AnalyticsPipelineConfig) -> dict[str, str]:
    paths = {"source": config.source, "taxonomy": config.taxonomy, "rules": config.rules}
    if config.create_pdf:
        paths["font"] = config.font
    for name in ("ml_corpus_manifest", "ml_reassignment_manifest"):
        path = getattr(config, name)
        if path is not None:
            paths[name] = path
    repository = Path(__file__).resolve().parents[2]
    for module in (
        "analytics_pipeline",
        "feedback_preparation",
        "feedback_loading",
        "feedback_reconciliation",
        "feedback_ml_adaptation",
        "problem_assessment",
        "problem_loading",
        "datalens_package",
        "analytical_report",
        "problem_priorities",
    ):
        paths[f"code:{module}"] = repository / "src/operations" / f"{module}.py"
    for path in (repository / "src/analytics").glob("*.py"):
        paths[f"code:analytics/{path.name}"] = path
    paths["dashboard_spec"] = repository / "configs/datalens-dashboard.v1.json"
    if any(not path.is_file() for path in paths.values()):
        msg = "pipeline input, configuration, or font file is missing"
        raise AnalyticsPipelineError(msg)
    return {name: _sha256(path) for name, path in paths.items()}


async def preflight_analytics_pipeline(
    config: AnalyticsPipelineConfig,
    factory: async_sessionmaker[AsyncSession],
) -> dict[str, object]:
    """Read prerequisites without creating files, loading rows, or applying migrations."""
    hashes = _input_hashes(config)
    if config.create_pdf and find_spec("reportlab") is None:
        msg = "PDF execution requires the reporting extra"
        raise AnalyticsPipelineError(msg)
    async with factory() as session, session.begin():
        await session.execute(text("SET TRANSACTION READ ONLY"))
        ready = await session.scalar(
            text("""
                SELECT to_regclass('analytics.problem_categories') IS NOT NULL
                   AND to_regclass('analytics_bi.category_overview') IS NOT NULL
                   AND to_regclass('analytics_bi.organization_monthly') IS NOT NULL
            """)
        )
    if not ready:
        msg = "analytics migrations must be applied through e1f2a3b4c5d6 before execution"
        raise AnalyticsPipelineError(msg)
    return {"input_sha256": hashes, "mode": "hybrid" if config.ml_corpus_manifest else "rules_only"}


def _save_state(path: Path, state: dict[str, object]) -> None:
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as target:
        target.write(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(path)


def _fingerprints(directory: Path) -> dict[str, str]:
    return {str(path.relative_to(directory)): _sha256(path) for path in sorted(directory.rglob("*")) if path.is_file()}


def _archive_attempt(directory: Path, root: Path) -> None:
    if directory.exists():
        archive = root / "attempts"
        archive.mkdir(exist_ok=True)
        target = archive / (directory.name + "-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f"))
        directory.rename(target)


def _bind_id(state: dict[str, object], key: str, value: int) -> None:
    if key in state and state[key] != value:
        msg = "database snapshot identity changed; start a new pipeline run"
        raise AnalyticsPipelineError(msg)
    state[key] = value


async def _bind_problem_content(
    factory: async_sessionmaker[AsyncSession],
    state: dict[str, object],
    run_id: int,
) -> None:
    digest = hashlib.sha256()
    async with factory() as session:
        rows = await session.stream(
            text("""
            SELECT feedback_id, category_id, stance, category_probability, stance_probability, method,
                   evidence_spans, model_name, model_revision, review_status, reviewer_id, assessed_at
            FROM analytics.problem_assessments WHERE run_id = :run ORDER BY feedback_id, category_id
        """),
            {"run": run_id},
        )
        async for row in rows:
            digest.update((json.dumps(list(row), default=str, ensure_ascii=False) + "\n").encode())
    current = digest.hexdigest()
    if "problem_content_sha256" in state and state["problem_content_sha256"] != current:
        msg = "stored problem assessments changed; use a new pipeline run for updated reports"
        raise AnalyticsPipelineError(msg)
    state["problem_content_sha256"] = current


async def _execute_stage(
    stage: AnalyticsStage,
    directory: Path,
    root: Path,
    config: AnalyticsPipelineConfig,
    factory: async_sessionmaker[AsyncSession],
    state: dict[str, object],
) -> None:
    feedback = root / "01-prepare/feedback.jsonl"
    feedback_manifest = root / "01-prepare/feedback.manifest.json"
    assessments = root / "05-assess/assessments.jsonl"
    if stage == AnalyticsStage.PREPARE:
        prepare_yandex_feedback(config.source, directory / "feedback.jsonl", directory / "feedback.manifest.json")
    elif stage == AnalyticsStage.LOAD_FEEDBACK:
        async with factory() as session:
            result = await load_feedback_snapshot(session, feedback, feedback_manifest)
        _bind_id(state, "dataset_id", result.dataset_id)
    elif stage == AnalyticsStage.RECONCILE:
        async with factory() as session:
            report = await reconcile_feedback_snapshot(session, feedback, feedback_manifest)
        if not report.is_valid:
            msg = "database feedback content does not match the canonical artifact"
            raise AnalyticsPipelineError(msg)
        (directory / "reconciliation.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    elif stage == AnalyticsStage.ADAPT_ML:
        if config.ml_corpus_manifest is not None and config.ml_reassignment_manifest is not None:
            from src.operations.feedback_ml_adaptation import adapt_feedback_ml_results  # noqa: PLC0415

            adapt_feedback_ml_results(
                feedback,
                feedback_manifest,
                config.ml_corpus_manifest,
                config.ml_reassignment_manifest,
                directory / "topics.jsonl",
            )
    elif stage == AnalyticsStage.ASSESS:
        topics = root / "04-adapt_ml/topics.jsonl" if config.ml_corpus_manifest else None
        assess_feedback_problems(
            feedback,
            feedback_manifest,
            config.taxonomy,
            config.rules,
            directory / "assessments.jsonl",
            topic_assignments_path=topics,
            topic_manifest_path=topics.with_suffix(".jsonl.manifest.json") if topics else None,
        )
    elif stage == AnalyticsStage.LOAD_PROBLEMS:
        async with factory() as session:
            loaded = await load_problem_assessments(
                session, assessments, assessments.with_suffix(".jsonl.manifest.json"), taxonomy_path=config.taxonomy
            )
        _bind_id(state, "problem_run_id", loaded.run_id)
        await _bind_problem_content(factory, state, loaded.run_id)
    elif stage == AnalyticsStage.DATALENS:
        build_datalens_package(directory / "package", int(str(state["problem_run_id"])))
    elif stage == AnalyticsStage.PDF and config.create_pdf:
        async with factory() as session:
            data = await load_analytical_report(session, int(str(state["problem_run_id"])))
        write_analytical_report(data, directory / "report", config.font)


async def run_analytics_pipeline(
    config: AnalyticsPipelineConfig,
    factory: async_sessionmaker[AsyncSession],
    *,
    resume: bool = False,
    stop_after: AnalyticsStage | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Execute stages with a single-writer lock, immutable inputs, and verified checkpoints."""
    preflight = await preflight_analytics_pipeline(config, factory)
    root = config.runs_root / config.run_name
    state_path = root / "pipeline-state.json"
    signature = {"config": config.model_dump(mode="json"), "inputs": preflight["input_sha256"]}
    if not resume:
        root.mkdir(parents=True, mode=0o700, exist_ok=False)
    lock = root / ".pipeline.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        msg = "pipeline is locked; verify no other process is running before removing its lock"
        raise AnalyticsPipelineError(msg) from None
    os.close(descriptor)
    initialized = False
    try:
        if resume:
            if not state_path.is_file():
                msg = "resume requires an existing pipeline state"
                raise AnalyticsPipelineError(msg)
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("signature") != signature:
                msg = "resume inputs or configuration changed; start a new run"
                raise AnalyticsPipelineError(msg)
        else:
            state = {"schema_version": 1, "signature": signature, "completed": {}, "status": "running"}
        initialized = True
        _save_state(state_path, state)
        completed = state["completed"]
        for index, stage in enumerate(AnalyticsStage, start=1):
            directory = root / f"{index:02d}-{stage.value}"
            checkpoint = completed.get(stage.value)
            if checkpoint is not None and _fingerprints(directory) != checkpoint:
                msg = f"completed artifacts changed at stage {stage.value}"
                raise AnalyticsPipelineError(msg)
            database_stage = stage in {
                AnalyticsStage.LOAD_FEEDBACK,
                AnalyticsStage.RECONCILE,
                AnalyticsStage.LOAD_PROBLEMS,
            }
            if checkpoint is None or database_stage:
                if progress:
                    progress(f"{index:02d} {stage.value}: {'verify' if checkpoint is not None else 'started'}")
                if checkpoint is None:
                    _archive_attempt(directory, root)
                directory.mkdir(exist_ok=True)
                state.update(status="running", current_stage=stage.value)
                _save_state(state_path, state)
                await _execute_stage(stage, directory, root, config, factory, state)
                if _input_hashes(config) != preflight["input_sha256"]:
                    msg = "source or configuration changed during execution"
                    raise AnalyticsPipelineError(msg)
                completed[stage.value] = _fingerprints(directory)
                _save_state(state_path, state)
            elif progress:
                progress(f"{index:02d} {stage.value}: verified checkpoint")
            if stage == stop_after:
                state["status"] = "partial"
                _save_state(state_path, state)
                return dict(state)
        state.update(status="completed", current_stage=None)
        _save_state(state_path, state)
        return dict(state)
    except BaseException:
        if initialized:
            state["status"] = "failed"
            _save_state(state_path, state)
        raise
    finally:
        lock.unlink(missing_ok=True)
