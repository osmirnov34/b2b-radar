import hashlib
import json
from pathlib import Path

import pytest

from src.ml.experiment_passport import (
    ExperimentPassportConfig,
    build_experiment_passport,
    write_experiment_passport,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _passport_run(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    stage = run / "01-inspection"
    stage.mkdir(parents=True)
    config = run / "pipeline-config.json"
    config.write_text('{"secret_path": "/private/input"}\n', encoding="utf-8")
    marker = stage / "inspection-manifest.json"
    marker.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "config": {"expected_records": 10},
                "created_at": "2025-01-01T00:00:00Z",
            },
        ),
        encoding="utf-8",
    )
    log = stage / "stage.log"
    log.write_text("completed\n", encoding="utf-8")
    pipeline = {
        "schema_version": 1,
        "run_id": "passport-run",
        "status": "awaiting_review",
        "source_dataset_sha256": "a" * 64,
        "config_snapshot_path": str(config),
        "config_snapshot_sha256": _sha256(config),
        "stages": [
            {
                "stage": "inspection",
                "status": "completed",
                "marker_path": str(marker),
                "marker_sha256": _sha256(marker),
                "log_path": str(log),
                "log_sha256": _sha256(log),
                "duration_seconds": 1.25,
                "return_code": 0,
            },
        ],
    }
    (run / "run-manifest.json").write_text(json.dumps(pipeline), encoding="utf-8")
    return run


def test_experiment_passport_records_versions_hashes_and_verified_stage_evidence(tmp_path: Path) -> None:
    run = _passport_run(tmp_path)
    config = ExperimentPassportConfig(packages=("pydantic", "definitely-not-installed-b2b-radar"))

    passport = build_experiment_passport(run, tmp_path / "not-a-repository", code_ref="reviewed-sha", config=config)

    assert passport.run_id == "passport-run"
    assert passport.code_ref == "reviewed-sha"
    assert passport.git_commit == "unknown"
    assert passport.git_dirty is None
    assert passport.completed_stages == 1
    assert passport.total_stage_duration_seconds == 1.25
    assert passport.stages[0].marker_path == "01-inspection/inspection-manifest.json"
    assert passport.manifests[0].config_sha256 is not None
    assert passport.environment.packages["pydantic"] is not None
    assert passport.environment.packages["definitely-not-installed-b2b-radar"] is None
    assert passport.private_source_values_included is False
    assert "/private/input" not in passport.model_dump_json()


def test_experiment_passport_writes_atomically_outside_run(tmp_path: Path) -> None:
    run = _passport_run(tmp_path)
    passport = build_experiment_passport(run, tmp_path, config=ExperimentPassportConfig(packages=("pydantic",)))
    output = tmp_path / "passports" / passport.run_id

    path = write_experiment_passport(passport, output, run)

    assert path.is_file()
    assert json.loads(path.read_text())["run_id"] == "passport-run"
    with pytest.raises(FileExistsError, match="already exists"):
        write_experiment_passport(passport, output, run)
    with pytest.raises(ValueError, match="outside"):
        write_experiment_passport(passport, run / "passport", run)


def test_experiment_passport_rejects_tampered_stage_or_configuration(tmp_path: Path) -> None:
    run = _passport_run(tmp_path)
    (run / "01-inspection/stage.log").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="log checksum mismatch"):
        build_experiment_passport(run, tmp_path)

    run = _passport_run(tmp_path / "second")
    (run / "pipeline-config.json").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="configuration snapshot checksum mismatch"):
        build_experiment_passport(run, tmp_path)


def test_experiment_passport_configuration_rejects_duplicate_packages() -> None:
    with pytest.raises(ValueError, match="unique"):
        ExperimentPassportConfig(packages=("numpy", "numpy"))
