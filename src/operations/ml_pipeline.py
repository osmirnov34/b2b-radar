"""Resumable orchestration and atomic publication of the offline ML pipeline."""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, ConfigDict, Field

from src.ml.evaluation import EvaluationMetrics, EvaluationStatus
from src.web.ml_snapshot import load_public_snapshot

if TYPE_CHECKING:
    from collections.abc import Sequence

_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")


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


class PipelineStatus(StrEnum):
    RUNNING = "running"
    PARTIAL = "partial"
    AWAITING_REVIEW = "awaiting_review"
    FAILED = "failed"
    COMPLETED = "completed"


class StageStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"


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


def _preflight(config: PipelineConfig, project_root: Path, run_dir: Path) -> None:
    if sys.version_info < (3, 11) or sys.version_info >= (3, 13):
        msg = "ML pipeline requires Python >=3.11,<3.13"
        raise RuntimeError(msg)
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
    missing = [str(path) for path in required if not path.is_file()]
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
    resolved_root = project_root.resolve()
    run_id = config.run_id or (run_dir.name if run_dir is not None else _new_run_id())
    if not _RUN_ID_PATTERN.fullmatch(run_id):
        msg = "run_id contains unsafe characters"
        raise ValueError(msg)
    active_run_dir = (run_dir or config.runs_root / run_id).resolve()
    _preflight(config, resolved_root, active_run_dir)
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
        created_at = manifest.created_at
        config_path = active_run_dir / f"pipeline-config-resume-{time.time_ns()}.json"
        config_path.write_text(f"{config.model_dump_json(indent=2)}\n", encoding="utf-8")
    else:
        config_path.write_text(f"{config.model_dump_json(indent=2)}\n", encoding="utf-8")
        records = []
        created_at = now
    config_hash = _sha256_file(config_path)
    stage_dirs = _paths(active_run_dir)
    markers = _markers(stage_dirs)
    completed = {record.stage: record for record in records if record.status == StageStatus.COMPLETED}
    if resume:
        for stage, record in completed.items():
            marker = Path(record.marker_path)
            if not marker.is_file() or record.marker_sha256 != _sha256_file(marker):
                msg = f"cannot resume: completed stage {stage.value!r} has a changed or missing marker"
                raise ValueError(msg)
    restart_index = list(PipelineStage).index(restart_from) if restart_from is not None else None
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
