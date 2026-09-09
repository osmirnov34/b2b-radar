"""Reproducible, privacy-safe passports for persisted ML pipeline runs."""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _PassportModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExperimentPassportConfig(_PassportModel):
    """Select stable dependency versions recorded for reproducibility."""

    schema_version: int = 1
    packages: tuple[str, ...] = (
        "hdbscan",
        "numpy",
        "pandas",
        "plotly",
        "pydantic",
        "scikit-learn",
        "sentence-transformers",
        "torch",
        "umap-learn",
    )

    @field_validator("packages")
    @classmethod
    def validate_packages(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or len(set(value)) != len(value) or any(not package.strip() for package in value):
            msg = "passport packages must be non-empty, unique names"
            raise ValueError(msg)
        return value


class ManifestEvidence(_PassportModel):
    """Reference one immutable stage manifest without copying its configuration."""

    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    schema_version: int | None = Field(default=None, ge=1)
    config_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    created_at: str | None = None


class RuntimeEnvironment(_PassportModel):
    """Record portable runtime properties without host or account identifiers."""

    python_version: str
    python_implementation: str
    operating_system: str
    machine: str
    packages: dict[str, str | None]


class StageEvidence(_PassportModel):
    """Record verified execution evidence for one attempted pipeline stage."""

    stage: str
    status: str
    marker_path: str
    marker_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    log_path: str
    log_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    duration_seconds: float = Field(ge=0)
    return_code: int


class ExperimentPassport(_PassportModel):
    """Describe exactly which code, data, configuration and artifacts formed a run."""

    passport_schema_version: int = 1
    run_id: str
    pipeline_status: str
    pipeline_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    config_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    code_ref: str
    git_commit: str
    git_dirty: bool | None
    environment: RuntimeEnvironment
    manifests: tuple[ManifestEvidence, ...]
    stages: tuple[StageEvidence, ...]
    completed_stages: int = Field(ge=0)
    total_stage_duration_seconds: float = Field(ge=0)
    private_source_values_included: bool = False
    generated_at: datetime


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _git_metadata(project_root: Path) -> tuple[str, bool | None]:
    git_executable = shutil.which("git")
    if git_executable is None:
        return "unknown", None
    try:
        commit = subprocess.run(  # noqa: S603 - resolved executable and fixed arguments
            [git_executable, "-C", str(project_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(  # noqa: S603 - resolved executable and fixed arguments
                [git_executable, "-C", str(project_root), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown", None
    else:
        return commit, dirty


def _manifest_evidence(path: Path, run_dir: Path) -> ManifestEvidence:
    if path.is_symlink():
        msg = f"experiment manifest cannot be a symbolic link: {path.name}"
        raise ValueError(msg)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        msg = f"experiment manifest must contain a JSON object: {path.name}"
        raise TypeError(msg)
    schema_version = data.get("schema_version")
    config = data.get("config")
    return ManifestEvidence(
        path=path.relative_to(run_dir).as_posix(),
        sha256=_sha256(path),
        schema_version=schema_version if isinstance(schema_version, int) else None,
        config_sha256=_canonical_sha256(config) if isinstance(config, dict) else None,
        created_at=str(data["created_at"]) if data.get("created_at") is not None else None,
    )


def _declared_run_path(run_dir: Path, raw_path: str) -> Path:
    declared = Path(raw_path)
    path = declared.resolve() if declared.is_absolute() else (run_dir / declared).resolve()
    if not path.is_relative_to(run_dir) or path.is_symlink():
        msg = f"experiment evidence escapes the selected run: {path.name}"
        raise ValueError(msg)
    return path


def _stage_evidence(stage: object, run_dir: Path) -> StageEvidence:
    if not isinstance(stage, dict):
        msg = "pipeline stage evidence must be a JSON object"
        raise TypeError(msg)
    marker = _declared_run_path(run_dir, str(stage["marker_path"]))
    log = _declared_run_path(run_dir, str(stage["log_path"]))
    marker_sha256 = stage.get("marker_sha256")
    if str(stage["status"]) == "completed" and (not marker.is_file() or marker_sha256 != _sha256(marker)):
        msg = f"completed stage marker checksum mismatch: {stage['stage']}"
        raise ValueError(msg)
    if not log.is_file() or stage["log_sha256"] != _sha256(log):
        msg = f"stage log checksum mismatch: {stage['stage']}"
        raise ValueError(msg)
    return StageEvidence(
        stage=str(stage["stage"]),
        status=str(stage["status"]),
        marker_path=marker.relative_to(run_dir).as_posix(),
        marker_sha256=str(marker_sha256) if marker_sha256 is not None else None,
        log_path=log.relative_to(run_dir).as_posix(),
        log_sha256=str(stage["log_sha256"]),
        duration_seconds=float(stage["duration_seconds"]),
        return_code=int(stage["return_code"]),
    )


def build_experiment_passport(
    run_dir: Path,
    project_root: Path,
    *,
    code_ref: str = "unknown",
    config: ExperimentPassportConfig | None = None,
) -> ExperimentPassport:
    """Build a passport from immutable manifests and the current code environment."""
    active_config = config or ExperimentPassportConfig()
    resolved_run = run_dir.resolve()
    pipeline_path = resolved_run / "run-manifest.json"
    if pipeline_path.is_symlink() or not pipeline_path.is_file():
        msg = "pipeline run manifest is missing or unsafe"
        raise FileNotFoundError(msg)
    pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
    config_snapshot = _declared_run_path(resolved_run, str(pipeline["config_snapshot_path"]))
    if _sha256(config_snapshot) != pipeline["config_snapshot_sha256"]:
        msg = "pipeline configuration snapshot checksum mismatch"
        raise ValueError(msg)
    manifest_paths = sorted(resolved_run.rglob("*-manifest.json"))
    manifests = tuple(_manifest_evidence(path, resolved_run) for path in manifest_paths)
    commit, dirty = _git_metadata(project_root.resolve())
    raw_stages = pipeline.get("stages", [])
    if not isinstance(raw_stages, list):
        msg = "pipeline stages must be a JSON array"
        raise TypeError(msg)
    stages = tuple(_stage_evidence(stage, resolved_run) for stage in raw_stages)
    environment = RuntimeEnvironment(
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
        operating_system=platform.system(),
        machine=platform.machine(),
        packages={package: _package_version(package) for package in active_config.packages},
    )
    return ExperimentPassport(
        run_id=str(pipeline["run_id"]),
        pipeline_status=str(pipeline["status"]),
        pipeline_manifest_sha256=_sha256(pipeline_path),
        source_dataset_sha256=str(pipeline["source_dataset_sha256"]),
        config_snapshot_sha256=str(pipeline["config_snapshot_sha256"]),
        code_ref=code_ref,
        git_commit=commit,
        git_dirty=dirty,
        environment=environment,
        manifests=manifests,
        stages=stages,
        completed_stages=sum(stage.status == "completed" for stage in stages),
        total_stage_duration_seconds=sum(stage.duration_seconds for stage in stages),
        generated_at=datetime.now(UTC),
    )


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def write_experiment_passport(
    passport: ExperimentPassport,
    output_dir: Path,
    source_run_dir: Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically persist a privacy-safe passport outside the immutable run tree."""
    target = output_dir.resolve()
    if target.is_relative_to(source_run_dir.resolve()):
        msg = "experiment passport must be stored outside the immutable run"
        raise ValueError(msg)
    path = target / "experiment-passport.json"
    if path.exists() and not overwrite:
        msg = f"experiment passport already exists: {path}"
        raise FileExistsError(msg)
    target.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(f"{passport.model_dump_json(indent=2)}\n", encoding="utf-8")
    temporary.replace(path)
    return path
