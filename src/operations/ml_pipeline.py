"""Resumable orchestration and atomic publication of the offline ML pipeline."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.ml.clustering import ClusteringManifest, HDBSCANConfig
from src.ml.config import CleaningConfig, DeduplicationConfig, EmbeddingConfig
from src.ml.dimensionality_reduction import UMAPConfig
from src.ml.evaluation import EvaluationConfig, EvaluationMetrics, EvaluationStatus
from src.ml.export import ExportConfig
from src.ml.inspection import DatasetFormat, detect_dataset_format
from src.ml.outlier_reassignment import OutlierReassignmentConfig
from src.ml.schemas import ExportedComment
from src.ml.topic_representation import TopicRepresentationConfig
from src.web.ml_snapshot import load_public_snapshot

if TYPE_CHECKING:
    from collections.abc import Sequence

_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
_DRY_RUN_STRUCTURE_SAMPLE = 100
_EXPECTED_RECORDS_TOLERANCE = 0.10
_COMMAND_PREFIX_LENGTH = 2
_MINIMUM_COMMAND_LENGTH = 3
_MINIMUM_SMOKE_RECORDS = 20


class _PipelineModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PipelineStage(StrEnum):
    INSPECTION = "inspection"
    SPLIT = "split"
    EDA = "eda"
    CLEANING = "cleaning"
    EMBEDDINGS = "embeddings"
    DEDUPLICATION = "deduplication"
    CORPUS = "corpus"
    REDUCTION = "reduction"
    CLUSTERING = "clustering"
    TOPICS = "topics"
    REASSIGNMENT = "reassignment"
    EVALUATION = "evaluation"
    EXPORT = "export"


_SMOKE_LAST_STAGE = PipelineStage.REASSIGNMENT
_SMOKE_STAGE_COUNT = list(PipelineStage).index(_SMOKE_LAST_STAGE) + 1


class PipelineStatus(StrEnum):
    RUNNING = "running"
    PARTIAL = "partial"
    AWAITING_REVIEW = "awaiting_review"
    FAILED = "failed"
    COMPLETED = "completed"


class StageStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"


class DryRunStatus(StrEnum):
    READY = "ready"
    WARNING = "warning"
    BLOCKED = "blocked"


class DryRunStageAction(StrEnum):
    RUN = "run"
    SKIP = "skip"
    RESTART = "restart"
    BLOCKED = "blocked"


class DryRunCheck(_PipelineModel):
    code: str = Field(pattern=r"^[a-z0-9_.-]+$")
    status: DryRunStatus
    message: str
    path: str | None = None


class DryRunDatasetSummary(_PipelineModel):
    path: str
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    lines_total: int = Field(ge=0)
    non_empty_records: int = Field(ge=0)
    expected_records: int | None = Field(default=None, ge=1)
    expected_format: str
    detected_format: str
    format_matches: bool
    usable: bool
    structural_errors: int = Field(ge=0)


class DryRunResourceEstimate(_PipelineModel):
    records: int = Field(ge=0)
    assumed_embedding_dimensions: int = Field(ge=1)
    embeddings_gb: float = Field(ge=0)
    estimated_working_disk_gb: float = Field(ge=0)
    total_memory_gb: float | None = Field(default=None, ge=0)
    gpu: str


class DryRunStageResult(_PipelineModel):
    number: int = Field(ge=1)
    stage: PipelineStage
    action: DryRunStageAction
    reason: str
    command: list[str]
    input_paths: list[str]
    output_paths: list[str]
    upstream_generated_inputs: list[str] = Field(default_factory=list)
    replaced_paths: list[str] = Field(default_factory=list)
    requires_force: bool = False
    blocking_checks: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class PipelineDryRunReport(_PipelineModel):
    schema_version: int = 1
    status: DryRunStatus
    run_id: str
    run_dir: str
    dataset: DryRunDatasetSummary | None
    available_disk_gb: float | None = Field(default=None, ge=0)
    resources: DryRunResourceEstimate | None = None
    checks: list[DryRunCheck]
    stages: list[DryRunStageResult]
    awaiting_review_expected: bool
    real_command: list[str]

    @property
    def can_run(self) -> bool:
        """Return whether preflight found no condition that must block execution."""
        return self.status != DryRunStatus.BLOCKED


def render_dry_run_report(report: PipelineDryRunReport, *, verbose: bool = False) -> str:
    """Render a privacy-safe report for terminals and notebook output."""
    symbols = {
        DryRunStatus.READY: "[READY]",
        DryRunStatus.WARNING: "[WARN]",
        DryRunStatus.BLOCKED: "[BLOCKED]",
    }
    counts = Counter(check.status for check in report.checks)
    lines = [
        f"Dry-run: {symbols[report.status]}",
        f"Run: {report.run_id}",
        f"Directory: {report.run_dir}",
        (
            "Checks: "
            f"{counts[DryRunStatus.READY]} ready, "
            f"{counts[DryRunStatus.WARNING]} warnings, "
            f"{counts[DryRunStatus.BLOCKED]} blocked"
        ),
    ]
    if report.dataset is not None:
        lines.append(
            "Dataset: "
            f"{report.dataset.non_empty_records} records; "
            f"{report.dataset.detected_format}; sha256={report.dataset.sha256[:12]}...",
        )
    if report.available_disk_gb is not None:
        lines.append(f"Free disk: {report.available_disk_gb:.2f} GiB")
    if report.resources is not None:
        memory = (
            f"{report.resources.total_memory_gb:.2f} GiB"
            if report.resources.total_memory_gb is not None
            else "unknown"
        )
        lines.append(
            "Resources: "
            f"embeddings≈{report.resources.embeddings_gb:.2f} GiB; "
            f"working disk≈{report.resources.estimated_working_disk_gb:.2f} GiB; "
            f"RAM={memory}; GPU={report.resources.gpu}",
        )
    lines.append("Stages:")
    lines.extend(
        f"  {stage.number:02d}. {stage.stage.value}: {stage.action.value}"
        + (f" ({'; '.join(stage.warnings)})" if stage.warnings else "")
        for stage in report.stages
    )
    visible_checks = [
        check
        for check in report.checks
        if verbose or check.status != DryRunStatus.READY
    ]
    if visible_checks:
        lines.append("Checks requiring attention:" if not verbose else "Checks:")
        lines.extend(
            f"  {symbols[check.status]} {check.code}: {check.message}"
            + (f" [{check.path}]" if check.path else "")
            for check in visible_checks
        )
    lines.append("Decision: execution allowed" if report.can_run else "Decision: execution blocked")
    lines.append(f"Command: {shlex.join(report.real_command)}")
    return "\n".join(lines)


class PipelineConfig(_PipelineModel):
    schema_version: int = 1
    source_dataset: Path
    runs_root: Path = Path("data/ml-runs")
    run_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
    python_executable: Path = Path(sys.executable)
    minimum_free_gb: float = Field(default=5.0, ge=0)
    expected_records: int | None = Field(default=None, ge=1)
    cleaning_config: Path = Path("configs/dataset-cleaning.example.json")
    embeddings_config: Path = Path("configs/embeddings.example.json")
    deduplication_config: Path = Path("configs/semantic-deduplication.example.json")
    umap_config: Path = Path("configs/umap.example.json")
    clustering_config: Path = Path("configs/hdbscan.example.json")
    topics_config: Path = Path("configs/topic-representation.example.json")
    reassignment_config: Path = Path("configs/outlier-reassignment.example.json")
    evaluation_config: Path = Path("configs/evaluation.example.json")
    export_config: Path = Path("configs/export.example.json")
    manual_annotations: Path | None = None


class StageRecord(_PipelineModel):
    stage: PipelineStage
    status: StageStatus
    command: list[str]
    marker_path: str
    marker_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    log_path: str
    log_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    return_code: int
    duration_seconds: float = Field(ge=0)
    finished_at: datetime


class PipelineRunManifest(_PipelineModel):
    schema_version: int = 1
    run_id: str
    status: PipelineStatus
    source_dataset_path: str
    source_dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    config_snapshot_path: str
    config_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    stages: list[StageRecord]
    current_stage: PipelineStage | None = None
    message: str = ""
    created_at: datetime
    updated_at: datetime


class SmokeSampleManifest(_PipelineModel):
    schema_version: int = 1
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sample_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    seed: int
    requested_records: int = Field(ge=1)
    selected_records: int = Field(ge=1)
    selected_groups: int = Field(ge=1)
    publishable: bool = False


class SmokeRunReport(_PipelineModel):
    schema_version: int = 1
    status: DryRunStatus
    workspace: str
    sample: SmokeSampleManifest
    pipeline_status: PipelineStatus
    checks: list[DryRunCheck]
    full_run_command: list[str]

    @property
    def full_run_allowed(self) -> bool:
        return self.status != DryRunStatus.BLOCKED


def render_smoke_run_report(report: SmokeRunReport) -> str:
    """Render the final smoke gate without including sampled text or provenance."""
    lines = [
        f"Smoke-run: [{report.status.value.upper()}]",
        f"Workspace: {report.workspace}",
        (
            f"Sample: {report.sample.selected_records} records in "
            f"{report.sample.selected_groups} complete groups; sha256={report.sample.sample_sha256[:12]}..."
        ),
        f"Pipeline status: {report.pipeline_status.value}",
        "Artifact checks:",
    ]
    lines.extend(f"  [{check.status.value.upper()}] {check.code}: {check.message}" for check in report.checks)
    lines.append("Full-run decision: allowed" if report.full_run_allowed else "Full-run decision: blocked")
    if report.full_run_allowed:
        lines.append(f"Command: {shlex.join(report.full_run_command)}")
    return "\n".join(lines)


class PipelineContext(_PipelineModel):
    run_id: str
    project_root: Path
    run_dir: Path
    stage_dirs: dict[PipelineStage, Path]
    markers: dict[PipelineStage, Path]
    resume: bool
    restart_from: PipelineStage | None = None
    stop_after: PipelineStage | None = None


class StageExecutor(Protocol):
    def __call__(self, command: Sequence[str], log_path: Path, cwd: Path) -> int: ...


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, model: BaseModel) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(f"{model.model_dump_json(indent=2)}\n", encoding="utf-8")
    temporary.replace(path)


def _default_executor(command: Sequence[str], log_path: Path, cwd: Path) -> int:
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(  # noqa: S603 -- argv is generated internally without a shell.
            list(command),
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    return completed.returncode


def _paths(run_dir: Path) -> dict[PipelineStage, Path]:
    return {stage: run_dir / f"{index:02d}-{stage.value}" for index, stage in enumerate(PipelineStage, start=1)}


def _markers(stage_dirs: dict[PipelineStage, Path]) -> dict[PipelineStage, Path]:
    return {
        PipelineStage.INSPECTION: stage_dirs[PipelineStage.INSPECTION] / "dataset-profile.json",
        PipelineStage.SPLIT: stage_dirs[PipelineStage.SPLIT] / "split-manifest.json",
        PipelineStage.EDA: stage_dirs[PipelineStage.EDA] / "development-profile.json",
        PipelineStage.CLEANING: stage_dirs[PipelineStage.CLEANING] / "cleaning-manifest.json",
        PipelineStage.EMBEDDINGS: stage_dirs[PipelineStage.EMBEDDINGS] / "embedding-manifest.json",
        PipelineStage.DEDUPLICATION: (
            stage_dirs[PipelineStage.DEDUPLICATION] / "semantic-deduplication-manifest.json"
        ),
        PipelineStage.CORPUS: stage_dirs[PipelineStage.CORPUS] / "corpus-manifest.json",
        PipelineStage.REDUCTION: stage_dirs[PipelineStage.REDUCTION] / "clustering-manifest.json",
        PipelineStage.CLUSTERING: stage_dirs[PipelineStage.CLUSTERING] / "clustering-manifest.json",
        PipelineStage.TOPICS: stage_dirs[PipelineStage.TOPICS] / "topic-representation-manifest.json",
        PipelineStage.REASSIGNMENT: (
            stage_dirs[PipelineStage.REASSIGNMENT] / "outlier-reassignment-manifest.json"
        ),
        PipelineStage.EVALUATION: stage_dirs[PipelineStage.EVALUATION] / "evaluation-manifest.json",
        PipelineStage.EXPORT: stage_dirs[PipelineStage.EXPORT] / "export-manifest.json",
    }


def _pipeline_context(
    config: PipelineConfig,
    project_root: Path,
    *,
    run_dir: Path | None,
    resume: bool,
    restart_from: PipelineStage | None,
    stop_after: PipelineStage | None,
) -> PipelineContext:
    run_id = config.run_id or (run_dir.name if run_dir is not None else _new_run_id())
    if not _RUN_ID_PATTERN.fullmatch(run_id):
        msg = "run_id contains unsafe characters"
        raise ValueError(msg)
    active_run_dir = (run_dir or config.runs_root / run_id).resolve()
    stage_dirs = _paths(active_run_dir)
    return PipelineContext(
        run_id=run_id,
        project_root=project_root.resolve(),
        run_dir=active_run_dir,
        stage_dirs=stage_dirs,
        markers=_markers(stage_dirs),
        resume=resume,
        restart_from=restart_from,
        stop_after=stop_after,
    )


def _command(
    stage: PipelineStage,
    config: PipelineConfig,
    project_root: Path,
    stage_dirs: dict[PipelineStage, Path],
    *,
    force: bool,
) -> list[str]:
    python = str(config.python_executable)
    def script(name: str) -> str:
        return str(project_root / "scripts" / name)
    dirs = stage_dirs
    split = dirs[PipelineStage.SPLIT]
    cleaning = dirs[PipelineStage.CLEANING]
    embeddings = dirs[PipelineStage.EMBEDDINGS]
    dedup = dirs[PipelineStage.DEDUPLICATION]
    corpus = dirs[PipelineStage.CORPUS]
    reduction = dirs[PipelineStage.REDUCTION]
    clustering = dirs[PipelineStage.CLUSTERING]
    topics = dirs[PipelineStage.TOPICS]
    reassignment = dirs[PipelineStage.REASSIGNMENT]
    evaluation = dirs[PipelineStage.EVALUATION]
    commands = {
        PipelineStage.INSPECTION: [
            python,
            script("inspect_dataset.py"),
            str(config.source_dataset),
            "--report-dir",
            str(dirs[stage]),
        ],
        PipelineStage.SPLIT: [
            python,
            script("split_dataset.py"),
            str(config.source_dataset),
            "--output-dir",
            str(split),
        ],
        PipelineStage.EDA: [
            python,
            script("run_eda.py"),
            str(split / "development.jsonl"),
            "--manifest",
            str(split / "split-manifest.json"),
            "--report-dir",
            str(dirs[stage]),
        ],
        PipelineStage.CLEANING: [
            python,
            script("clean_dataset.py"),
            str(split / "development.jsonl"),
            "--manifest",
            str(split / "split-manifest.json"),
            "--config",
            str(config.cleaning_config),
            "--output-dir",
            str(cleaning),
        ],
        PipelineStage.EMBEDDINGS: [
            python,
            script("generate_embeddings.py"),
            str(cleaning / "development-clean.jsonl"),
            "--config",
            str(config.embeddings_config),
            "--output-dir",
            str(embeddings),
        ],
        PipelineStage.DEDUPLICATION: [
            python,
            script("semantic_deduplicate.py"),
            str(cleaning / "development-clean.jsonl"),
            str(embeddings / "embeddings.npy"),
            "--embedding-manifest",
            str(embeddings / "embedding-manifest.json"),
            "--config",
            str(config.deduplication_config),
            "--output-dir",
            str(dedup),
        ],
        PipelineStage.CORPUS: [
            python,
            script("build_corpus.py"),
            str(cleaning / "development-clean.jsonl"),
            str(embeddings / "embeddings.npy"),
            str(dedup / "keep-indices.json"),
            "--cleaning-manifest",
            str(cleaning / "cleaning-manifest.json"),
            "--embedding-manifest",
            str(embeddings / "embedding-manifest.json"),
            "--groups",
            str(dedup / "semantic-groups.jsonl"),
            "--deduplication-manifest",
            str(dedup / "semantic-deduplication-manifest.json"),
            "--output-dir",
            str(corpus),
        ],
        PipelineStage.REDUCTION: [
            python,
            script("reduce_dimensions.py"),
            str(corpus / "final-embeddings.npy"),
            "--corpus-manifest",
            str(corpus / "corpus-manifest.json"),
            "--config",
            str(config.umap_config),
            "--output-dir",
            str(reduction),
            "--mode",
            "clustering",
        ],
        PipelineStage.CLUSTERING: [
            python,
            script("cluster_corpus.py"),
            str(reduction / "clustering-reduced.npy"),
            "--reduction-manifest",
            str(reduction / "clustering-manifest.json"),
            "--corpus-manifest",
            str(corpus / "corpus-manifest.json"),
            "--config",
            str(config.clustering_config),
            "--output-dir",
            str(clustering),
        ],
        PipelineStage.TOPICS: [
            python,
            script("build_topic_representations.py"),
            str(corpus / "final-corpus.jsonl"),
            str(clustering / "cluster-labels.npy"),
            "--embeddings",
            str(corpus / "final-embeddings.npy"),
            "--probabilities",
            str(clustering / "cluster-probabilities.npy"),
            "--corpus-manifest",
            str(corpus / "corpus-manifest.json"),
            "--clustering-manifest",
            str(clustering / "clustering-manifest.json"),
            "--config",
            str(config.topics_config),
            "--output-dir",
            str(topics),
        ],
        PipelineStage.REASSIGNMENT: [
            python,
            script("reassign_outliers.py"),
            str(corpus / "final-embeddings.npy"),
            str(clustering / "cluster-labels.npy"),
            "--probabilities",
            str(clustering / "cluster-probabilities.npy"),
            "--corpus",
            str(corpus / "final-corpus.jsonl"),
            "--corpus-manifest",
            str(corpus / "corpus-manifest.json"),
            "--clustering-manifest",
            str(clustering / "clustering-manifest.json"),
            "--topic-manifest",
            str(topics / "topic-representation-manifest.json"),
            "--config",
            str(config.reassignment_config),
            "--output-dir",
            str(reassignment),
        ],
        PipelineStage.EVALUATION: [
            python,
            script("evaluate_topics.py"),
            str(corpus / "final-embeddings.npy"),
            str(clustering / "cluster-labels.npy"),
            str(reassignment / "final-cluster-labels.npy"),
            str(reassignment / "final-cluster-confidence.npy"),
            "--corpus",
            str(corpus / "final-corpus.jsonl"),
            "--corpus-manifest",
            str(corpus / "corpus-manifest.json"),
            "--clustering-manifest",
            str(clustering / "clustering-manifest.json"),
            "--topic-manifest",
            str(topics / "topic-representation-manifest.json"),
            "--reassignment-manifest",
            str(reassignment / "outlier-reassignment-manifest.json"),
            "--config",
            str(config.evaluation_config),
            "--output-dir",
            str(evaluation),
        ],
        PipelineStage.EXPORT: [
            python,
            script("export_topic_results.py"),
            "--corpus-manifest",
            str(corpus / "corpus-manifest.json"),
            "--clustering-manifest",
            str(clustering / "clustering-manifest.json"),
            "--topic-manifest",
            str(topics / "topic-representation-manifest.json"),
            "--reassignment-manifest",
            str(reassignment / "outlier-reassignment-manifest.json"),
            "--evaluation-manifest",
            str(evaluation / "evaluation-manifest.json"),
            "--config",
            str(config.export_config),
            "--output-dir",
            str(dirs[stage]),
        ],
    }
    command = commands[stage]
    if stage == PipelineStage.INSPECTION and config.expected_records is not None:
        command.extend(["--expected-records", str(config.expected_records)])
    if stage == PipelineStage.EVALUATION and config.manual_annotations is not None:
        command.extend(["--manual-annotations", str(config.manual_annotations)])
    if force and stage != PipelineStage.INSPECTION:
        command.append("--force")
    return command


def _required_files(config: PipelineConfig) -> list[Path]:
    required = [
        config.source_dataset,
        config.python_executable,
        config.cleaning_config,
        config.embeddings_config,
        config.deduplication_config,
        config.umap_config,
        config.clustering_config,
        config.topics_config,
        config.reassignment_config,
        config.evaluation_config,
        config.export_config,
    ]
    if config.manual_annotations is not None:
        required.append(config.manual_annotations)
    return required


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path.resolve()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _available_disk_gb(path: Path) -> float | None:
    parent = _nearest_existing_parent(path)
    if not parent.exists():
        return None
    return shutil.disk_usage(parent).free / 1024**3


def _check(
    code: str,
    passed: bool,
    success: str,
    failure: str,
    *,
    path: Path | None = None,
    warning: bool = False,
) -> DryRunCheck:
    status = DryRunStatus.READY if passed else (DryRunStatus.WARNING if warning else DryRunStatus.BLOCKED)
    return DryRunCheck(
        code=code,
        status=status,
        message=success if passed else failure,
        path=str(path) if path else None,
    )


def _configuration_checks(config: PipelineConfig) -> tuple[list[DryRunCheck], dict[str, BaseModel]]:
    specifications: tuple[tuple[str, Path, type[BaseModel]], ...] = (
        ("cleaning", config.cleaning_config, CleaningConfig),
        ("embeddings", config.embeddings_config, EmbeddingConfig),
        ("deduplication", config.deduplication_config, DeduplicationConfig),
        ("umap", config.umap_config, UMAPConfig),
        ("clustering", config.clustering_config, HDBSCANConfig),
        ("topics", config.topics_config, TopicRepresentationConfig),
        ("reassignment", config.reassignment_config, OutlierReassignmentConfig),
        ("evaluation", config.evaluation_config, EvaluationConfig),
        ("export", config.export_config, ExportConfig),
    )
    checks: list[DryRunCheck] = []
    validated: dict[str, BaseModel] = {}
    for name, path, model in specifications:
        try:
            value = model.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValidationError, ValueError):
            checks.append(
                DryRunCheck(
                    code=f"config_validation.{name}",
                    status=DryRunStatus.BLOCKED,
                    message="configuration failed strict typed validation",
                    path=str(path),
                ),
            )
            continue
        validated[name] = value
        schema_version = getattr(value, "schema_version", None)
        checks.append(
            _check(
                f"config_validation.{name}",
                schema_version == 1,
                "configuration passed strict schema-version-1 validation",
                "configuration uses an unsupported schema version",
                path=path,
            ),
        )
    embeddings = validated.get("embeddings")
    if isinstance(embeddings, EmbeddingConfig):
        checks.extend(
            [
                _check(
                    "compatibility.normalized_embeddings",
                    embeddings.normalize,
                    "normalized embeddings are compatible with cosine-based downstream stages",
                    "downstream cosine stages require normalized embeddings",
                    path=config.embeddings_config,
                ),
                _check(
                    "reproducibility.embedding_revision",
                    embeddings.model_revision is not None,
                    "embedding model revision is pinned",
                    "embedding model revision is not pinned",
                    path=config.embeddings_config,
                    warning=True,
                ),
            ],
        )
    deduplication = validated.get("deduplication")
    if isinstance(deduplication, DeduplicationConfig):
        exhaustive_safe = not (
            deduplication.backend == "exhaustive"
            and config.expected_records is not None
            and config.expected_records > deduplication.exhaustive_max_records
        )
        checks.append(
            _check(
                "compatibility.deduplication_scale",
                exhaustive_safe,
                "deduplication backend is compatible with the expected dataset scale",
                "exhaustive deduplication limit is below the expected record count",
                path=config.deduplication_config,
            ),
        )
    umap = validated.get("umap")
    if isinstance(umap, UMAPConfig):
        checks.append(
            _check(
                "reproducibility.umap_threads",
                umap.threads == 1,
                "UMAP uses one thread with a fixed random seed",
                "reproducible UMAP requires threads=1",
                path=config.umap_config,
            ),
        )
    clustering = validated.get("clustering")
    if isinstance(clustering, HDBSCANConfig):
        checks.append(
            _check(
                "compatibility.clustering_role",
                clustering.dataset_role == "development",
                "clustering is restricted to the development role",
                "clustering must not tune against validation or test data",
                path=config.clustering_config,
            ),
        )
    evaluation = validated.get("evaluation")
    if isinstance(evaluation, EvaluationConfig):
        checks.extend(
            [
                _check(
                    "production.evaluation_validation",
                    evaluation.validation_completed,
                    "validation evaluation is marked complete",
                    "validation evaluation is not complete",
                    path=config.evaluation_config,
                    warning=True,
                ),
                _check(
                    "production.evaluation_bootstrap",
                    evaluation.bootstrap_runs > 0,
                    "bootstrap stability evaluation is enabled",
                    "bootstrap stability evaluation is disabled",
                    path=config.evaluation_config,
                    warning=True,
                ),
                _check(
                    "production.manual_annotations",
                    config.manual_annotations is not None,
                    "manual annotation file is configured",
                    "manual annotation file is not configured",
                    path=config.manual_annotations,
                    warning=True,
                ),
            ],
        )
    export = validated.get("export")
    if isinstance(export, ExportConfig):
        checks.extend(
            [
                _check(
                    "security.export_research_text",
                    not export.include_research_text,
                    "public export excludes research text",
                    "public pipeline export must not include research text",
                    path=config.export_config,
                ),
                _check(
                    "production.export_final_evaluation",
                    export.require_final_evaluation,
                    "export requires a final passing evaluation",
                    "export does not require a final passing evaluation",
                    path=config.export_config,
                    warning=True,
                ),
            ],
        )
    return checks, validated


def _environment_checks(
    config: PipelineConfig,
    context: PipelineContext,
) -> tuple[list[DryRunCheck], float | None]:
    checks = [
        _check(
            "python.version",
            (3, 11) <= sys.version_info[:2] < (3, 13),
            f"active Python {sys.version_info.major}.{sys.version_info.minor} is supported",
            "active Python must be >=3.11,<3.13",
        ),
        _check(
            "python.executable",
            config.python_executable.is_file() and os.access(config.python_executable, os.R_OK | os.X_OK),
            "configured Python executable is readable and executable",
            "configured Python executable is missing or not executable",
            path=config.python_executable,
        ),
    ]
    config_file_checks = (
        ("config.cleaning", config.cleaning_config),
        ("config.embeddings", config.embeddings_config),
        ("config.deduplication", config.deduplication_config),
        ("config.umap", config.umap_config),
        ("config.clustering", config.clustering_config),
        ("config.topics", config.topics_config),
        ("config.reassignment", config.reassignment_config),
        ("config.evaluation", config.evaluation_config),
        ("config.export", config.export_config),
    )
    required_checks: tuple[tuple[str, Path], ...] = (("dataset.source", config.source_dataset), *config_file_checks)
    if config.manual_annotations is not None:
        required_checks = (*required_checks, ("annotations.manual", config.manual_annotations))
    for label, path in required_checks:
        checks.append(
            _check(
                label,
                path.is_file() and os.access(path, os.R_OK),
                "required input is readable",
                "required input is missing or unreadable",
                path=path,
            ),
        )
    for script_name in _SCRIPT_NAMES:
        script_path = context.project_root / "scripts" / script_name
        checks.append(
            _check(
                f"script.{script_path.stem}",
                script_path.is_file() and os.access(script_path, os.R_OK),
                "pipeline CLI is available",
                "pipeline CLI is missing or unreadable",
                path=script_path,
            ),
        )
    optional_modules = {
        "numpy": "numpy",
        "sentence_transformers": "sentence_transformers",
        "umap": "umap",
        "hdbscan": "hdbscan",
        "bertopic": "bertopic",
        "sklearn": "sklearn",
        "scipy": "scipy",
    }
    for code, module in optional_modules.items():
        checks.append(
            _check(
                f"dependency.{code}",
                importlib.util.find_spec(module) is not None,
                "analysis dependency is installed",
                "analysis dependency is not installed",
            ),
        )
    writable_parent = _nearest_existing_parent(context.run_dir)
    checks.append(
        _check(
            "filesystem.runs_root",
            writable_parent.is_dir() and os.access(writable_parent, os.W_OK | os.X_OK),
            "nearest existing run parent is writable",
            "no writable existing parent is available for the run directory",
            path=writable_parent,
        ),
    )
    available = _available_disk_gb(context.run_dir)
    checks.append(
        _check(
            "filesystem.free_space",
            available is not None and available >= config.minimum_free_gb,
            f"available disk space satisfies the {config.minimum_free_gb:.1f} GiB minimum",
            f"available disk space is below the {config.minimum_free_gb:.1f} GiB minimum",
            path=writable_parent,
        ),
    )
    return checks, available


def _dataset_dry_run(config: PipelineConfig) -> tuple[DryRunDatasetSummary | None, list[DryRunCheck]]:
    path = config.source_dataset
    if not path.is_file() or not os.access(path, os.R_OK):
        return None, []
    try:
        format_result = detect_dataset_format(path)
        digest = hashlib.sha256()
        lines_total = 0
        non_empty_records = 0
        sampled = 0
        structural_errors = 0
        with path.open("rb") as source:
            for raw_line in source:
                digest.update(raw_line)
                lines_total += 1
                if not raw_line.strip():
                    continue
                non_empty_records += 1
                if sampled >= _DRY_RUN_STRUCTURE_SAMPLE:
                    continue
                sampled += 1
                try:
                    value = json.loads(raw_line.decode("utf-8-sig"))
                    ExportedComment.model_validate(value)
                except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                    structural_errors += 1
    except OSError:
        return None, [
            DryRunCheck(
                code="dataset.inspection",
                status=DryRunStatus.BLOCKED,
                message="dataset inspection failed without exposing record content",
                path=str(path),
            ),
        ]
    expected_matches = True
    if config.expected_records is not None:
        difference = abs(non_empty_records - config.expected_records) / config.expected_records
        expected_matches = difference <= _EXPECTED_RECORDS_TOLERANCE
    usable = format_result.matches and structural_errors == 0 and expected_matches
    summary = DryRunDatasetSummary(
        path=str(path),
        size_bytes=path.stat().st_size,
        sha256=digest.hexdigest(),
        lines_total=lines_total,
        non_empty_records=non_empty_records,
        expected_records=config.expected_records,
        expected_format=format_result.expected.value,
        detected_format=format_result.detected.value,
        format_matches=format_result.matches,
        usable=usable,
        structural_errors=structural_errors,
    )
    checks = [
        _check(
            "dataset.format",
            format_result.matches and format_result.detected == DatasetFormat.JSONL,
            "dataset matches the expected JSONL format",
            f"expected JSONL but detected {format_result.detected.value}",
            path=path,
        ),
        _check(
            "dataset.contract",
            usable,
            "sampled dataset structure and record-count expectation are usable",
            "dataset structure or record-count expectation blocks the pipeline",
            path=path,
        ),
    ]
    return summary, checks


def _command_path_tokens(command: list[str]) -> list[str]:
    output_value_indices = {
        index + 1
        for index, token in enumerate(command[:-1])
        if token in {"--output-dir", "--report-dir"}
    }
    paths = []
    for index, token in enumerate(command):
        if index in output_value_indices or token.startswith("-"):
            continue
        candidate = Path(token)
        if index < _COMMAND_PREFIX_LENGTH or candidate.suffix or "/" in token:
            paths.append(token)
    return paths


def _command_checks(
    stage: PipelineStage,
    command: list[str],
    context: PipelineContext,
    config: PipelineConfig,
) -> list[DryRunCheck]:
    flags = [token for token in command if token.startswith("--")]
    duplicate_flags = sorted(flag for flag, count in Counter(flags).items() if count > 1)
    unsafe = any(
        token in {"&&", "||", ";", "|", ">", ">>", "<"} or "$(" in token or "`" in token
        for token in command
    )
    output_flag = "--report-dir" if stage in {PipelineStage.INSPECTION, PipelineStage.EDA} else "--output-dir"
    required_flags = {
        PipelineStage.INSPECTION: {"--report-dir"},
        PipelineStage.SPLIT: {"--output-dir"},
        PipelineStage.EDA: {"--manifest", "--report-dir"},
        PipelineStage.CLEANING: {"--manifest", "--config", "--output-dir"},
        PipelineStage.EMBEDDINGS: {"--config", "--output-dir"},
        PipelineStage.DEDUPLICATION: {"--embedding-manifest", "--config", "--output-dir"},
        PipelineStage.CORPUS: {
            "--cleaning-manifest",
            "--embedding-manifest",
            "--groups",
            "--deduplication-manifest",
            "--output-dir",
        },
        PipelineStage.REDUCTION: {"--corpus-manifest", "--config", "--output-dir", "--mode"},
        PipelineStage.CLUSTERING: {"--reduction-manifest", "--corpus-manifest", "--config", "--output-dir"},
        PipelineStage.TOPICS: {
            "--embeddings",
            "--probabilities",
            "--corpus-manifest",
            "--clustering-manifest",
            "--config",
            "--output-dir",
        },
        PipelineStage.REASSIGNMENT: {
            "--probabilities",
            "--corpus",
            "--corpus-manifest",
            "--clustering-manifest",
            "--topic-manifest",
            "--config",
            "--output-dir",
        },
        PipelineStage.EVALUATION: {
            "--corpus",
            "--corpus-manifest",
            "--clustering-manifest",
            "--topic-manifest",
            "--reassignment-manifest",
            "--config",
            "--output-dir",
        },
        PipelineStage.EXPORT: {
            "--corpus-manifest",
            "--clustering-manifest",
            "--topic-manifest",
            "--reassignment-manifest",
            "--evaluation-manifest",
            "--config",
            "--output-dir",
        },
    }[stage]
    allowed_external = {
        path.resolve()
        for path in (
            *_required_files(config),
            *(context.project_root / "scripts" / name for name in _SCRIPT_NAMES),
        )
    }
    escaped_paths = [
        path
        for token in _command_path_tokens(command)
        if not (path := Path(token).resolve()).is_relative_to(context.run_dir) and path not in allowed_external
    ]
    return [
        _check(
            f"command.{stage.value}.shape",
            len(command) >= _MINIMUM_COMMAND_LENGTH and command[0] == str(Path(command[0])) and output_flag in command,
            "command contains executable, script, arguments, and output destination",
            "command is missing a required executable, script, or output destination",
        ),
        _check(
            f"command.{stage.value}.flags",
            not duplicate_flags and required_flags.issubset(flags),
            "command contains every required flag with no duplicate singleton flags",
            (
                f"command flags are incomplete or duplicated: {', '.join(duplicate_flags)}"
                if duplicate_flags
                else "command is missing one or more required flags"
            ),
        ),
        _check(
            f"command.{stage.value}.argv",
            not unsafe,
            "command is a shell-free argv vector",
            "command contains a forbidden shell control token",
        ),
        _check(
            f"command.{stage.value}.marker_scope",
            context.markers[stage].resolve().is_relative_to(context.run_dir),
            "expected stage marker stays inside the run directory",
            "expected stage marker escapes the run directory",
            path=context.markers[stage],
        ),
        _check(
            f"command.{stage.value}.input_scope",
            not escaped_paths,
            "command inputs are run-local or explicitly allowed external files",
            "command contains an input path outside the run and allowed external files",
        ),
    ]


def _run_scope_checks(config: PipelineConfig, context: PipelineContext) -> list[DryRunCheck]:
    runs_root = config.runs_root.resolve()
    in_scope = context.run_dir.is_relative_to(runs_root)
    checks = [
        _check(
            "run.scope",
            in_scope,
            "run directory stays inside runs_root",
            "run directory escapes runs_root or resolves through an external symlink",
            path=context.run_dir,
        ),
        _check(
            "restart.requires_resume",
            context.restart_from is None or context.resume,
            "restart-from usage is compatible with resume mode",
            "restart-from is allowed only together with resume",
        ),
    ]
    if context.run_dir.exists():
        temporary = [path for path in context.run_dir.rglob("*") if path.name.endswith(".tmp")]
        symlinks = [path for path in context.run_dir.rglob("*") if path.is_symlink()]
        checks.extend(
            [
                _check(
                    "run.temporary_files",
                    not temporary,
                    "run directory contains no temporary files",
                    "run directory contains unfinished temporary files",
                    path=context.run_dir,
                ),
                _check(
                    "run.partial_symlinks",
                    not symlinks,
                    "run directory contains no unexpected symlinks",
                    "run directory contains symlinks and requires manual review",
                    path=context.run_dir,
                ),
            ],
        )
    return checks


def _preflight(config: PipelineConfig, project_root: Path, run_dir: Path) -> None:
    if sys.version_info < (3, 11) or sys.version_info >= (3, 13):
        msg = "ML pipeline requires Python >=3.11,<3.13"
        raise RuntimeError(msg)
    missing = [str(path) for path in _required_files(config) if not path.is_file()]
    if missing:
        msg = f"pipeline preflight found missing files: {', '.join(missing)}"
        raise FileNotFoundError(msg)
    scripts = [project_root / "scripts" / name for name in _SCRIPT_NAMES]
    missing_scripts = [str(path) for path in scripts if not path.is_file()]
    if missing_scripts:
        msg = f"pipeline scripts are missing: {', '.join(missing_scripts)}"
        raise FileNotFoundError(msg)
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(run_dir.parent).free / 1024**3
    if free_gb < config.minimum_free_gb:
        msg = f"pipeline requires {config.minimum_free_gb:.1f} GiB free; found {free_gb:.1f} GiB"
        raise RuntimeError(msg)


_SCRIPT_NAMES = (
    "inspect_dataset.py",
    "split_dataset.py",
    "run_eda.py",
    "clean_dataset.py",
    "generate_embeddings.py",
    "semantic_deduplicate.py",
    "build_corpus.py",
    "reduce_dimensions.py",
    "cluster_corpus.py",
    "build_topic_representations.py",
    "reassign_outliers.py",
    "evaluate_topics.py",
    "export_topic_results.py",
)


def _new_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _resume_plan(
    config: PipelineConfig,
    context: PipelineContext,
) -> tuple[dict[PipelineStage, StageRecord], PipelineStage | None, list[DryRunCheck]]:
    if not context.resume:
        check = _check(
            "run.directory",
            not context.run_dir.exists(),
            "new run directory is available",
            "new run directory already exists",
            path=context.run_dir,
        )
        return {}, None, [check]
    manifest_path = context.run_dir / "run-manifest.json"
    if not manifest_path.is_file():
        return {}, None, [
            DryRunCheck(
                code="resume.manifest",
                status=DryRunStatus.BLOCKED,
                message="resume requires an existing run manifest",
                path=str(manifest_path),
            ),
        ]
    try:
        manifest = PipelineRunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}, None, [
            DryRunCheck(
                code="resume.manifest",
                status=DryRunStatus.BLOCKED,
                message="run manifest failed validation without exposing its content",
                path=str(manifest_path),
            ),
        ]
    checks = [
        _check(
            "resume.run_id",
            manifest.run_id == context.run_id,
            "resume run ID matches",
            "resume run ID does not match",
            path=manifest_path,
        ),
        _check(
            "resume.dataset",
            config.source_dataset.is_file()
            and manifest.source_dataset_sha256 == _sha256_file(config.source_dataset),
            "source dataset checksum matches the recorded run",
            "source dataset checksum does not match the recorded run",
            path=config.source_dataset,
        ),
    ]
    expected_sequence = list(PipelineStage)[: len(manifest.stages)]
    actual_sequence = [record.stage for record in manifest.stages]
    sequence_valid = actual_sequence == expected_sequence
    checks.append(
        _check(
            "resume.stage_sequence",
            sequence_valid,
            "recorded stages form a unique contiguous prefix",
            "recorded stages contain a duplicate, gap, or ordering violation",
            path=manifest_path,
        ),
    )
    config_path = Path(manifest.config_snapshot_path)
    checks.append(
        _check(
            "resume.config_snapshot",
            config_path.is_file() and manifest.config_snapshot_sha256 == _sha256_file(config_path),
            "recorded configuration snapshot is intact",
            "recorded configuration snapshot changed or is missing",
            path=config_path,
        ),
    )
    completed: dict[PipelineStage, StageRecord] = {}
    failed_stage = next((record.stage for record in manifest.stages if record.status == StageStatus.FAILED), None)
    for record in manifest.stages if sequence_valid else []:
        if record.status != StageStatus.COMPLETED:
            continue
        marker = Path(record.marker_path)
        valid = marker.is_file() and record.marker_sha256 == _sha256_file(marker)
        checks.append(
            _check(
                f"resume.marker.{record.stage.value}",
                valid,
                "completed stage marker checksum matches",
                "completed stage marker changed or is missing",
                path=marker,
            ),
        )
        if valid:
            completed[record.stage] = record
    for stage in PipelineStage:
        stage_dir = context.stage_dirs[stage]
        if stage in completed or not stage_dir.exists():
            continue
        partial = any(stage_dir.iterdir())
        checks.append(
            _check(
                f"resume.partial.{stage.value}",
                not partial or stage == failed_stage,
                "stage contains no unrecorded partial artifacts",
                "stage contains unrecorded partial artifacts; restart-from is required",
                path=stage_dir,
            ),
        )
    return completed, failed_stage, checks


def _manual_gate_check(config: PipelineConfig) -> tuple[DryRunCheck, bool]:
    try:
        evaluation = EvaluationConfig.model_validate_json(config.evaluation_config.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return (
            DryRunCheck(
                code="evaluation.manual_gate",
                status=DryRunStatus.BLOCKED,
                message="evaluation configuration cannot be validated",
                path=str(config.evaluation_config),
            ),
            True,
        )
    awaiting = config.manual_annotations is None or not evaluation.validation_completed
    return (
        DryRunCheck(
            code="evaluation.manual_gate",
            status=DryRunStatus.WARNING if awaiting else DryRunStatus.READY,
            message=(
                "pipeline is expected to pause for manual review"
                if awaiting
                else "manual annotations and validation completion are configured"
            ),
            path=str(config.evaluation_config),
        ),
        awaiting,
    )


def _final_evaluation_passes(context: PipelineContext) -> bool:
    metrics_path = context.stage_dirs[PipelineStage.EVALUATION] / "evaluation-metrics.json"
    try:
        metrics = EvaluationMetrics.model_validate_json(metrics_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return metrics.status == EvaluationStatus.PASS and not metrics.preliminary


def _resource_estimate(config: PipelineConfig, records: int) -> DryRunResourceEstimate:
    """Return conservative, model-independent storage figures without initializing a GPU."""
    dimensions = 1024
    embeddings_gb = records * dimensions * 4 / 1024**3
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        total_memory_gb = pages * page_size / 1024**3
    except (OSError, ValueError):
        total_memory_gb = None
    try:
        embedding = EmbeddingConfig.model_validate_json(config.embeddings_config.read_text(encoding="utf-8"))
        requested_device = embedding.device
    except (OSError, ValueError):
        requested_device = "unknown"
    if requested_device == "cpu":
        gpu = "not requested"
    elif Path("/dev/nvidia0").exists():
        gpu = "NVIDIA device detected (capacity not probed)"
    elif importlib.util.find_spec("torch") is None:
        gpu = "not detected; torch unavailable"
    else:
        gpu = "not detected without device initialization"
    return DryRunResourceEstimate(
        records=records,
        assumed_embedding_dimensions=dimensions,
        embeddings_gb=embeddings_gb,
        estimated_working_disk_gb=max(embeddings_gb * 4, config.minimum_free_gb),
        total_memory_gb=total_memory_gb,
        gpu=gpu,
    )


def dry_run_pipeline(
    config: PipelineConfig,
    project_root: Path,
    *,
    config_path: Path | None = None,
    run_dir: Path | None = None,
    resume: bool = False,
    restart_from: PipelineStage | None = None,
    stop_after: PipelineStage | None = None,
) -> PipelineDryRunReport:
    """Build a full read-only execution plan without creating files or running processes."""
    context = _pipeline_context(
        config,
        project_root,
        run_dir=run_dir,
        resume=resume,
        restart_from=restart_from,
        stop_after=stop_after,
    )
    checks, available = _environment_checks(config, context)
    configuration_checks, _validated_configs = _configuration_checks(config)
    checks.extend(configuration_checks)
    dataset, dataset_checks = _dataset_dry_run(config)
    checks.extend(dataset_checks)
    resources = _resource_estimate(config, dataset.non_empty_records) if dataset is not None else None
    if resources is not None:
        disk_constrained = available is not None and available < resources.estimated_working_disk_gb
        memory_constrained = (
            resources.total_memory_gb is not None
            and resources.total_memory_gb < resources.embeddings_gb * 3
        )
        checks.extend(
            [
                DryRunCheck(
                    code="resources.estimated_disk",
                    status=DryRunStatus.WARNING if disk_constrained else DryRunStatus.READY,
                    message=(
                        "available disk is below the conservative working estimate"
                        if disk_constrained
                        else "available disk satisfies the conservative working estimate"
                    ),
                ),
                DryRunCheck(
                    code="resources.estimated_memory",
                    status=DryRunStatus.WARNING if memory_constrained else DryRunStatus.READY,
                    message=(
                        "system RAM is below three estimated embedding matrices; batching is required"
                        if memory_constrained
                        else "system RAM is compatible with the coarse embedding estimate"
                    ),
                ),
            ],
        )
    checks.extend(_run_scope_checks(config, context))
    completed, failed_stage, resume_checks = _resume_plan(config, context)
    checks.extend(resume_checks)
    gate_check, awaiting_review = _manual_gate_check(config)
    checks.append(gate_check)
    if restart_from == PipelineStage.EXPORT:
        checks.append(
            _check(
                "restart.export_evaluation",
                _final_evaluation_passes(context),
                "export restart has a final passing evaluation",
                "export cannot restart without a final passing evaluation",
                path=context.stage_dirs[PipelineStage.EVALUATION] / "evaluation-metrics.json",
            ),
        )
    stage_order = list(PipelineStage)
    if (
        restart_from is not None
        and stage_order.index(PipelineStage.CLEANING)
        <= stage_order.index(restart_from)
        <= stage_order.index(PipelineStage.REASSIGNMENT)
    ):
        checks.append(
            DryRunCheck(
                code="restart.manual_review",
                status=DryRunStatus.WARNING,
                message="restart changes model inputs or labels and requires renewed manual review",
                path=str(context.stage_dirs[restart_from]),
            ),
        )
    if resume:
        checks.append(
            DryRunCheck(
                code="resume.config_compatibility",
                status=DryRunStatus.READY,
                message="verified upstream markers remain protected by downstream manifest checksum contracts",
                path=str(context.run_dir / "run-manifest.json"),
            ),
        )
    for stage in PipelineStage:
        command = _command(stage, config, context.project_root, context.stage_dirs, force=False)
        checks.extend(_command_checks(stage, command, context, config))
    blocked_codes = [check.code for check in checks if check.status == DryRunStatus.BLOCKED]
    effective_restart = restart_from or failed_stage
    restart_index = stage_order.index(effective_restart) if effective_restart is not None else None
    stages = []
    for index, stage in enumerate(PipelineStage):
        force = restart_index is not None and index >= restart_index
        if stage in completed and not force:
            action = DryRunStageAction.SKIP
            reason = "completed marker checksum is verified"
        elif blocked_codes:
            action = DryRunStageAction.BLOCKED
            reason = "one or more read-only preflight checks failed"
        elif force:
            action = DryRunStageAction.RESTART
            reason = "selected by restart-from; existing outputs will require --force"
        else:
            action = DryRunStageAction.RUN
            reason = "no verified completed marker"
        command = _command(stage, config, context.project_root, context.stage_dirs, force=force)
        input_paths = sorted(set(_command_path_tokens(command)))
        upstream_generated = [
            path
            for path in input_paths
            if not Path(path).exists() and Path(path).resolve().is_relative_to(context.run_dir)
        ]
        replaced = []
        if action == DryRunStageAction.RESTART and context.stage_dirs[stage].exists():
            replaced = sorted(
                str(path)
                for path in context.stage_dirs[stage].rglob("*")
                if path.is_file() or path.is_symlink()
            )
        stage_warnings = []
        if action == DryRunStageAction.RESTART and stage == PipelineStage.INSPECTION:
            stage_warnings.append("inspection has no --force flag and rewrites only its report files")
        if action == DryRunStageAction.RESTART and stage in {
            PipelineStage.CLEANING,
            PipelineStage.EMBEDDINGS,
            PipelineStage.DEDUPLICATION,
            PipelineStage.CORPUS,
            PipelineStage.REDUCTION,
            PipelineStage.CLUSTERING,
            PipelineStage.TOPICS,
            PipelineStage.REASSIGNMENT,
        }:
            stage_warnings.append("downstream manual review must be repeated")
        stages.append(
            DryRunStageResult(
                number=index + 1,
                stage=stage,
                action=action,
                reason=reason,
                command=command,
                input_paths=input_paths,
                output_paths=[str(context.markers[stage])],
                upstream_generated_inputs=upstream_generated,
                replaced_paths=replaced,
                requires_force=force and stage != PipelineStage.INSPECTION,
                blocking_checks=blocked_codes,
                warnings=stage_warnings,
            ),
        )
        if stop_after == stage:
            break
    if blocked_codes:
        status = DryRunStatus.BLOCKED
    elif any(check.status == DryRunStatus.WARNING for check in checks):
        status = DryRunStatus.WARNING
    else:
        status = DryRunStatus.READY
    real_config = str(config_path) if config_path is not None else "<pipeline-config.json>"
    real_command = [str(config.python_executable), "scripts/run_ml_pipeline.py", "run", real_config]
    if run_dir is not None:
        real_command.extend(["--run-dir", str(run_dir)])
    if resume:
        real_command.append("--resume")
    if restart_from is not None:
        real_command.extend(["--restart-from", restart_from.value])
    if stop_after is not None:
        real_command.extend(["--stop-after", stop_after.value])
    return PipelineDryRunReport(
        status=status,
        run_id=context.run_id,
        run_dir=str(context.run_dir),
        dataset=dataset,
        available_disk_gb=available,
        resources=resources,
        checks=checks,
        stages=stages,
        awaiting_review_expected=awaiting_review,
        real_command=real_command,
    )


def _evaluation_status(stage_dirs: dict[PipelineStage, Path]) -> tuple[EvaluationStatus, bool]:
    path = stage_dirs[PipelineStage.EVALUATION] / "evaluation-metrics.json"
    metrics = EvaluationMetrics.model_validate_json(path.read_text(encoding="utf-8"))
    return metrics.status, metrics.preliminary


def run_pipeline(
    config: PipelineConfig,
    project_root: Path,
    *,
    run_dir: Path | None = None,
    resume: bool = False,
    restart_from: PipelineStage | None = None,
    stop_after: PipelineStage | None = None,
    executor: StageExecutor = _default_executor,
) -> PipelineRunManifest:
    """Run stages in order, preserving checksummed resume state after every process."""
    context = _pipeline_context(
        config,
        project_root,
        run_dir=run_dir,
        resume=resume,
        restart_from=restart_from,
        stop_after=stop_after,
    )
    if restart_from is not None and not resume:
        msg = "restart-from is allowed only together with resume"
        raise ValueError(msg)
    if not context.run_dir.is_relative_to(config.runs_root.resolve()):
        msg = "run directory must stay inside runs_root"
        raise ValueError(msg)
    resolved_root = context.project_root
    run_id = context.run_id
    active_run_dir = context.run_dir
    _preflight(config, context.project_root, context.run_dir)
    manifest_path = active_run_dir / "run-manifest.json"
    config_path = active_run_dir / "pipeline-config.json"
    if active_run_dir.exists() and not resume:
        msg = f"run directory already exists: {active_run_dir}"
        raise FileExistsError(msg)
    active_run_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC)
    if resume:
        manifest = PipelineRunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        if manifest.run_id != run_id or manifest.source_dataset_sha256 != _sha256_file(config.source_dataset):
            msg = "resume configuration does not match the existing run"
            raise ValueError(msg)
        previous_config = Path(manifest.config_snapshot_path)
        if not previous_config.is_file() or _sha256_file(previous_config) != manifest.config_snapshot_sha256:
            msg = "cannot resume: the recorded pipeline configuration changed or is missing"
            raise ValueError(msg)
        records = list(manifest.stages)
        if [record.stage for record in records] != list(PipelineStage)[: len(records)]:
            msg = "cannot resume: recorded stages are duplicated, missing, or out of order"
            raise ValueError(msg)
        created_at = manifest.created_at
        config_path = active_run_dir / f"pipeline-config-resume-{time.time_ns()}.json"
        config_path.write_text(f"{config.model_dump_json(indent=2)}\n", encoding="utf-8")
    else:
        config_path.write_text(f"{config.model_dump_json(indent=2)}\n", encoding="utf-8")
        records = []
        created_at = now
    config_hash = _sha256_file(config_path)
    stage_dirs = context.stage_dirs
    markers = context.markers
    completed = {record.stage: record for record in records if record.status == StageStatus.COMPLETED}
    if resume:
        for stage, record in completed.items():
            marker = Path(record.marker_path)
            if not marker.is_file() or record.marker_sha256 != _sha256_file(marker):
                msg = f"cannot resume: completed stage {stage.value!r} has a changed or missing marker"
                raise ValueError(msg)
    effective_restart = restart_from
    if effective_restart is None and records and records[-1].status == StageStatus.FAILED:
        effective_restart = records[-1].stage
    restart_index = list(PipelineStage).index(effective_restart) if effective_restart is not None else None
    if restart_index is not None:
        records = [record for record in records if list(PipelineStage).index(record.stage) < restart_index]
        completed = {record.stage: record for record in records}

    def persist(status: PipelineStatus, current: PipelineStage | None, message: str = "") -> PipelineRunManifest:
        state = PipelineRunManifest(
            run_id=run_id,
            status=status,
            source_dataset_path=str(config.source_dataset),
            source_dataset_sha256=_sha256_file(config.source_dataset),
            config_snapshot_path=str(config_path),
            config_snapshot_sha256=config_hash,
            stages=records,
            current_stage=current,
            message=message,
            created_at=created_at,
            updated_at=datetime.now(UTC),
        )
        _atomic_json(manifest_path, state)
        return state

    persist(PipelineStatus.RUNNING, None)
    for index, stage in enumerate(PipelineStage):
        if stage in completed:
            if stop_after == stage:
                return persist(PipelineStatus.PARTIAL, None, "partial run completed at requested stage")
            continue
        stage_dir = stage_dirs[stage]
        stage_dir.mkdir(parents=True, exist_ok=True)
        log_path = active_run_dir / "logs" / f"{index + 1:02d}-{stage.value}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        force = restart_index is not None and index >= restart_index
        command = _command(stage, config, resolved_root, stage_dirs, force=force)
        persist(PipelineStatus.RUNNING, stage)
        started = time.monotonic()
        return_code = executor(command, log_path, resolved_root)
        duration = time.monotonic() - started
        marker = markers[stage]
        marker_hash = _sha256_file(marker) if marker.is_file() else None
        record = StageRecord(
            stage=stage,
            status=StageStatus.COMPLETED if return_code == 0 and marker_hash is not None else StageStatus.FAILED,
            command=command,
            marker_path=str(marker),
            marker_sha256=marker_hash,
            log_path=str(log_path),
            log_sha256=_sha256_file(log_path),
            return_code=return_code,
            duration_seconds=duration,
            finished_at=datetime.now(UTC),
        )
        records.append(record)
        if record.status == StageStatus.FAILED:
            return persist(PipelineStatus.FAILED, stage, f"stage {stage.value} failed; see its local log")
        if stage == PipelineStage.EVALUATION:
            status, preliminary = _evaluation_status(stage_dirs)
            if preliminary:
                return persist(
                    PipelineStatus.AWAITING_REVIEW,
                    None,
                    "manual annotations and completed validation are required before export",
                )
            if status != EvaluationStatus.PASS:
                return persist(PipelineStatus.FAILED, stage, "evaluation did not pass publication thresholds")
        if stop_after == stage:
            return persist(PipelineStatus.PARTIAL, None, "partial run completed at requested stage")
    return persist(PipelineStatus.COMPLETED, None, "all pipeline stages completed")


def _smoke_group_key(comment: ExportedComment) -> str:
    """Match the splitter's provenance priority without exposing provenance in manifests."""
    if comment.video_id.strip():
        return f"video_id:{comment.video_id.strip()}"
    if comment.video_url.strip():
        return f"video_url:{comment.video_url.strip()}"
    channel_title = f"{comment.video_channel.strip()}\x1f{comment.video_title.strip()}"
    if channel_title != "\x1f":
        return f"channel_title:{channel_title}"
    if comment.comment_id.strip():
        return f"comment_id:{comment.comment_id.strip()}"
    return f"record:{hashlib.sha256(comment.model_dump_json().encode()).hexdigest()}"


def create_smoke_sample(
    source_path: Path,
    target_path: Path,
    *,
    records: int = 2_000,
    seed: int = 42,
) -> SmokeSampleManifest:
    """Create a deterministic whole-group JSONL sample for a local non-public smoke run."""
    if records < _MINIMUM_SMOKE_RECORDS:
        msg = "smoke sample must contain at least 20 requested records"
        raise ValueError(msg)
    if target_path.exists():
        msg = f"smoke sample already exists: {target_path}"
        raise FileExistsError(msg)
    counts: Counter[str] = Counter()
    source_hash = hashlib.sha256()
    with source_path.open("rb") as source:
        for line_number, raw_line in enumerate(source, start=1):
            source_hash.update(raw_line)
            if not raw_line.strip():
                continue
            try:
                comment = ExportedComment.model_validate_json(raw_line)
            except ValueError as exc:
                msg = f"invalid JSONL record at line {line_number}: {type(exc).__name__}"
                raise ValueError(msg) from exc
            counts[_smoke_group_key(comment)] += 1
    ranked = sorted(counts, key=lambda key: hashlib.sha256(f"{seed}\x1f{key}".encode()).digest())
    selected: set[str] = set()
    selected_records = 0
    for key in ranked:
        selected.add(key)
        selected_records += counts[key]
        if selected_records >= records:
            break
    if not selected:
        msg = "cannot create a smoke sample from an empty dataset"
        raise ValueError(msg)
    target_path.parent.mkdir(parents=True, exist_ok=False)
    sample_hash = hashlib.sha256()
    written = 0
    temporary = target_path.with_suffix(f"{target_path.suffix}.tmp")
    try:
        with source_path.open("rb") as source, temporary.open("wb") as target:
            for raw_line in source:
                if not raw_line.strip():
                    continue
                comment = ExportedComment.model_validate_json(raw_line)
                if _smoke_group_key(comment) not in selected:
                    continue
                target.write(raw_line)
                sample_hash.update(raw_line)
                written += 1
        temporary.replace(target_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return SmokeSampleManifest(
        source_sha256=source_hash.hexdigest(),
        sample_sha256=sample_hash.hexdigest(),
        seed=seed,
        requested_records=records,
        selected_records=written,
        selected_groups=len(selected),
    )


def _write_smoke_configs(config: PipelineConfig, directory: Path, records: int) -> PipelineConfig:
    directory.mkdir()
    embedding = EmbeddingConfig.model_validate_json(config.embeddings_config.read_text(encoding="utf-8"))
    umap = UMAPConfig.model_validate_json(config.umap_config.read_text(encoding="utf-8"))
    clustering = HDBSCANConfig.model_validate_json(config.clustering_config.read_text(encoding="utf-8"))
    embedding_path = directory / "embeddings.json"
    umap_path = directory / "umap.json"
    clustering_path = directory / "hdbscan.json"
    smoke_embedding = embedding.model_copy(update={"batch_size": min(embedding.batch_size, 16)})
    smoke_umap = umap.model_copy(
        update={
            "training_sample_size": records,
            "trustworthiness_sample_size": min(records, 500),
        },
    )
    smoke_clustering = clustering.model_copy(
        update={
            "min_cluster_size": max(2, min(clustering.min_cluster_size, records // _MINIMUM_SMOKE_RECORDS)),
            "allow_single_cluster": True,
            "dbcv_sample_size": 0,
        },
    )
    embedding_path.write_text(
        f"{smoke_embedding.model_dump_json(indent=2)}\n",
        encoding="utf-8",
    )
    umap_path.write_text(
        f"{smoke_umap.model_dump_json(indent=2)}\n",
        encoding="utf-8",
    )
    clustering_path.write_text(
        f"{smoke_clustering.model_dump_json(indent=2)}\n",
        encoding="utf-8",
    )
    return config.model_copy(
        update={
            "embeddings_config": embedding_path,
            "umap_config": umap_path,
            "clustering_config": clustering_path,
        },
    )


def _smoke_artifact_checks(run_dir: Path, manifest: PipelineRunManifest) -> list[DryRunCheck]:
    checks = []
    for record in manifest.stages:
        marker = Path(record.marker_path)
        valid = marker.is_file() and record.marker_sha256 == _sha256_file(marker)
        checks.append(
            _check(
                f"smoke.artifact.{record.stage.value}",
                valid,
                "stage marker exists and matches its manifest checksum",
                "stage marker is missing or has a checksum mismatch",
                path=marker,
            ),
        )
    temporary = [path for path in run_dir.rglob("*") if path.name.endswith(".tmp")]
    checks.append(
        _check(
            "smoke.temporary_files",
            not temporary,
            "smoke run left no incomplete temporary files",
            "smoke run left incomplete temporary files",
            path=run_dir,
        ),
    )
    try:
        arrays = [np.load(path, mmap_mode="r", allow_pickle=False) for path in run_dir.rglob("*.npy")]
        finite = all(bool(np.isfinite(array).all()) for array in arrays)
        lengths = {
            int(array.shape[0])
            for array in arrays
            if array.ndim >= 1
            and any(part in str(array.filename) for part in ("corpus", "clustering", "reassignment"))
        }
        aligned = len(lengths) <= 1
    except (OSError, ValueError):
        finite = False
        aligned = False
    checks.extend(
        [
            _check(
                "smoke.arrays_finite",
                finite,
                "numeric artifacts contain only finite values",
                "numeric artifacts are unreadable or contain NaN/Inf",
                path=run_dir,
            ),
            _check(
                "smoke.row_alignment",
                aligned,
                "corpus, clustering, and reassignment arrays have aligned rows",
                "downstream arrays disagree on their first dimension",
                path=run_dir,
            ),
        ],
    )
    clustering_manifest_path = run_dir / "09-clustering" / "clustering-manifest.json"
    try:
        clustering_manifest = ClusteringManifest.model_validate_json(
            clustering_manifest_path.read_text(encoding="utf-8"),
        )
    except (OSError, ValueError):
        pass
    else:
        excessive_outliers = (
            clustering_manifest.metrics.outlier_share
            > clustering_manifest.config.max_outlier_share_warning
        )
        checks.append(
            DryRunCheck(
                code="smoke.outlier_share",
                status=DryRunStatus.WARNING if excessive_outliers else DryRunStatus.READY,
                message=(
                    "small-sample outlier share exceeds its diagnostic threshold"
                    if excessive_outliers
                    else "small-sample outlier share is within its diagnostic threshold"
                ),
                path=str(clustering_manifest_path),
            ),
        )
    return checks


def run_smoke_pipeline(
    config: PipelineConfig,
    project_root: Path,
    *,
    records: int = 2_000,
    seed: int = 42,
    config_path: Path | None = None,
    executor: StageExecutor = _default_executor,
) -> SmokeRunReport:
    """Run stages 1-11 on a deterministic local sample; never evaluate or publish it."""
    source_preflight = dry_run_pipeline(config, project_root)
    if not source_preflight.can_run:
        msg = "source dry-run is blocked; smoke workspace was not created"
        raise RuntimeError(msg)
    run_id = config.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    smoke_run_id = f"{run_id[:74]}-smoke"
    workspace = (config.runs_root / smoke_run_id).resolve()
    if workspace.exists():
        msg = f"smoke workspace already exists: {workspace}"
        raise FileExistsError(msg)
    sample_path = workspace / "input" / "comments.jsonl"
    sample = create_smoke_sample(config.source_dataset, sample_path, records=records, seed=seed)
    smoke_config = _write_smoke_configs(config, workspace / "configs", sample.selected_records).model_copy(
        update={
            "source_dataset": sample_path,
            "expected_records": sample.selected_records,
            "run_id": smoke_run_id,
            "manual_annotations": None,
        },
    )
    smoke_config_path = workspace / "pipeline-smoke.json"
    smoke_config_path.write_text(f"{smoke_config.model_dump_json(indent=2)}\n", encoding="utf-8")
    (workspace / "NON_PUBLISHABLE").write_text(
        "Smoke-run artifacts are for pipeline verification only and must never be published.\n",
        encoding="utf-8",
    )
    run_dir = workspace / "run"
    preflight = dry_run_pipeline(smoke_config, project_root, config_path=smoke_config_path, run_dir=run_dir)
    if not preflight.can_run:
        msg = "smoke-run preflight is blocked; inspect dry-run output"
        raise RuntimeError(msg)
    manifest = run_pipeline(
        smoke_config,
        project_root,
        run_dir=run_dir,
        stop_after=_SMOKE_LAST_STAGE,
        executor=executor,
    )
    checks = _smoke_artifact_checks(run_dir, manifest)
    successful = manifest.status == PipelineStatus.PARTIAL and len(manifest.stages) == _SMOKE_STAGE_COUNT
    checks.append(
        _check(
            "smoke.stage_sequence",
            successful,
            "all automatic pre-review stages completed",
            "smoke run did not complete every automatic pre-review stage",
            path=run_dir / "run-manifest.json",
        ),
    )
    if any(item.status == DryRunStatus.BLOCKED for item in checks):
        status = DryRunStatus.BLOCKED
    elif any(item.status == DryRunStatus.WARNING for item in checks):
        status = DryRunStatus.WARNING
    else:
        status = DryRunStatus.READY
    return SmokeRunReport(
        status=status,
        workspace=str(workspace),
        sample=sample,
        pipeline_status=manifest.status,
        checks=checks,
        full_run_command=[
            str(config.python_executable),
            "scripts/run_ml_pipeline.py",
            "run",
            str(config_path) if config_path is not None else "<pipeline-config.json>",
        ],
    )


def publish_snapshot(export_dir: Path, publish_root: Path, run_id: str) -> Path:
    """Atomically switch the current symlink after strict public-snapshot validation."""
    if not _RUN_ID_PATTERN.fullmatch(run_id):
        msg = "run_id contains unsafe characters"
        raise ValueError(msg)
    resolved_export = export_dir.resolve()
    manifest_path = resolved_export / "export-manifest.json"
    load_public_snapshot(manifest_path, allow_unreliable=False)
    publish_root.mkdir(parents=True, exist_ok=True)
    current = publish_root / "current"
    previous = publish_root / "previous"
    temporary = publish_root / f".current-{run_id}.tmp"
    if temporary.exists() or temporary.is_symlink():
        msg = f"temporary publication link already exists: {temporary}"
        raise FileExistsError(msg)
    if current.exists() and not current.is_symlink():
        msg = "publication target 'current' exists and is not a managed symlink"
        raise FileExistsError(msg)
    previous_tmp = publish_root / f".previous-{run_id}.tmp"
    if current.is_symlink() and (previous_tmp.exists() or previous_tmp.is_symlink()):
        msg = f"temporary rollback link already exists: {previous_tmp}"
        raise FileExistsError(msg)
    temporary.symlink_to(resolved_export, target_is_directory=True)
    if current.is_symlink():
        prior_target = current.resolve()
        previous_tmp.symlink_to(prior_target, target_is_directory=True)
        previous_tmp.replace(previous)
    temporary.replace(current)
    return current / "export-manifest.json"


def rollback_snapshot(publish_root: Path) -> Path:
    """Atomically point current at the previously validated release without deleting data."""
    current = publish_root / "current"
    previous = publish_root / "previous"
    if not previous.is_symlink():
        msg = "no previous ML snapshot is available for rollback"
        raise FileNotFoundError(msg)
    if current.exists() and not current.is_symlink():
        msg = "rollback target 'current' exists and is not a managed symlink"
        raise FileExistsError(msg)
    target = previous.resolve()
    load_public_snapshot(target / "export-manifest.json", allow_unreliable=False)
    temporary = publish_root / ".rollback-current.tmp"
    if temporary.exists() or temporary.is_symlink():
        msg = f"temporary rollback link already exists: {temporary}"
        raise FileExistsError(msg)
    temporary.symlink_to(target, target_is_directory=True)
    temporary.replace(current)
    return current / "export-manifest.json"
