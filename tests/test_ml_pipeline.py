import json
import sys
from collections import Counter
from pathlib import Path

import pytest

from scripts import run_ml_pipeline as pipeline_cli
from src.ml.evaluation import (
    EvaluationMetrics,
    EvaluationStatus,
    GeometryMetrics,
    ManualMetrics,
)
from src.ml.export import ExportConfig, export_topic_results
from src.operations.ml_pipeline import (
    DryRunStageAction,
    DryRunStatus,
    PipelineConfig,
    PipelineStage,
    PipelineStatus,
    create_smoke_sample,
    dry_run_pipeline,
    publish_snapshot,
    render_dry_run_report,
    render_smoke_run_report,
    rollback_snapshot,
    run_pipeline,
    run_smoke_pipeline,
)
from tests.test_evaluation import _run_evaluation

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _config(tmp_path: Path, *, run_id: str = "test-run") -> PipelineConfig:
    source = tmp_path / "comments.jsonl"
    source.write_text('{"comment_text":"Useful CRM workflow","video_id":"v1"}\n')
    return PipelineConfig(
        source_dataset=source,
        runs_root=tmp_path / "runs",
        run_id=run_id,
        python_executable=Path(sys.executable),
        minimum_free_gb=0,
        expected_records=1,
    )


def _filesystem_snapshot(root: Path) -> dict[str, tuple[str, int, int, str | None]]:
    result = {}
    for path in sorted(root.rglob("*")):
        stat = path.lstat()
        kind = "symlink" if path.is_symlink() else ("file" if path.is_file() else "directory")
        target = str(path.readlink()) if path.is_symlink() else None
        result[str(path.relative_to(root))] = (kind, stat.st_size, stat.st_mtime_ns, target)
    return result


def _marker_for_script(script_name: str, output: Path) -> Path:
    names = {
        "inspect_dataset.py": "dataset-profile.json",
        "split_dataset.py": "split-manifest.json",
        "run_eda.py": "development-profile.json",
        "clean_dataset.py": "cleaning-manifest.json",
        "generate_embeddings.py": "embedding-manifest.json",
        "semantic_deduplicate.py": "semantic-deduplication-manifest.json",
        "build_corpus.py": "corpus-manifest.json",
        "reduce_dimensions.py": "clustering-manifest.json",
        "cluster_corpus.py": "clustering-manifest.json",
        "build_topic_representations.py": "topic-representation-manifest.json",
        "reassign_outliers.py": "outlier-reassignment-manifest.json",
        "evaluate_topics.py": "evaluation-manifest.json",
        "export_topic_results.py": "export-manifest.json",
    }
    return output / names[script_name]


def _successful_executor(command: list[str], log_path: Path, _cwd: Path) -> int:
    script_name = Path(command[1]).name
    flag = "--report-dir" if "--report-dir" in command else "--output-dir"
    output = Path(command[command.index(flag) + 1])
    output.mkdir(parents=True, exist_ok=True)
    _marker_for_script(script_name, output).write_text("{}\n")
    log_path.write_text(f"completed {script_name}\n")
    return 0


def _preliminary_evaluation_executor(command: list[str], log_path: Path, cwd: Path) -> int:
    result = _successful_executor(command, log_path, cwd)
    if Path(command[1]).name == "evaluate_topics.py":
        output = Path(command[command.index("--output-dir") + 1])
        geometry = GeometryMetrics(evaluated_records=1, clusters=1)
        metrics = EvaluationMetrics(
            records=1,
            topics=1,
            original_outlier_share=0,
            final_outlier_share=0,
            changed_label_share=0,
            original_geometry=geometry,
            final_geometry=geometry,
            bootstrap_runs=[],
            manual=ManualMetrics(
                annotations=0,
                sensitive_data_flags=0,
                merge_candidates=0,
                split_candidates=0,
            ),
            suspicious_topic_terms=0,
            status=EvaluationStatus.PASS_WITH_WARNINGS,
            preliminary=True,
        )
        (output / "evaluation-metrics.json").write_text(metrics.model_dump_json())
    return result


def test_pipeline_runs_partially_and_resumes_only_verified_stages(tmp_path: Path) -> None:
    config = _config(tmp_path)
    run_dir = tmp_path / "runs" / "test-run"
    partial = run_pipeline(
        config,
        PROJECT_ROOT,
        stop_after=PipelineStage.CLEANING,
        executor=_successful_executor,
    )
    assert partial.status == PipelineStatus.PARTIAL
    assert [record.stage for record in partial.stages] == list(PipelineStage)[:4]

    resumed = run_pipeline(
        config,
        PROJECT_ROOT,
        run_dir=run_dir,
        resume=True,
        stop_after=PipelineStage.REASSIGNMENT,
        executor=_successful_executor,
    )
    assert [record.stage for record in resumed.stages] == list(PipelineStage)[:11]
    assert len(list(run_dir.glob("pipeline-config*.json"))) == 2


def test_dry_run_is_read_only_and_builds_commands_without_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, run_id="dry-run")
    before = _filesystem_snapshot(tmp_path)

    def forbidden_subprocess(*_args: object, **_kwargs: object) -> None:
        msg = "dry-run must not launch subprocesses"
        raise AssertionError(msg)

    monkeypatch.setattr("src.operations.ml_pipeline.subprocess.run", forbidden_subprocess)
    report = dry_run_pipeline(config, PROJECT_ROOT)

    assert _filesystem_snapshot(tmp_path) == before
    assert not (tmp_path / "runs" / "dry-run").exists()
    assert report.dataset is not None
    assert report.dataset.detected_format == "jsonl"
    assert report.dataset.non_empty_records == 1
    assert report.awaiting_review_expected is True
    assert len(report.stages) == 13
    assert report.status in {DryRunStatus.READY, DryRunStatus.WARNING, DryRunStatus.BLOCKED}
    dedup = next(stage for stage in report.stages if stage.stage == PipelineStage.DEDUPLICATION)
    assert dedup.command.count("--output-dir") == 1
    assert "Useful CRM workflow" not in report.model_dump_json()


def test_dry_run_report_is_actionable_and_does_not_render_comment_text(tmp_path: Path) -> None:
    config = _config(tmp_path, run_id="rendered-dry-run")

    report = dry_run_pipeline(config, PROJECT_ROOT)
    rendered = render_dry_run_report(report)

    assert f"Dry-run: [{report.status.value.upper()}]" in rendered
    assert "Decision: execution " in rendered
    assert "01. inspection:" in rendered
    assert "scripts/run_ml_pipeline.py run" in rendered
    assert "Useful CRM workflow" not in rendered
    assert report.can_run is (report.status != DryRunStatus.BLOCKED)


def test_smoke_sample_is_deterministic_and_keeps_video_groups_whole(tmp_path: Path) -> None:
    source = tmp_path / "comments.jsonl"
    rows = [
        {"comment_text": f"Comment {index}", "comment_id": str(index), "video_id": f"v{index // 4}"}
        for index in range(40)
    ]
    source.write_text("".join(f"{json.dumps(row)}\n" for row in rows))

    first = create_smoke_sample(source, tmp_path / "first" / "sample.jsonl", records=20, seed=17)
    second = create_smoke_sample(source, tmp_path / "second" / "sample.jsonl", records=20, seed=17)

    assert first.sample_sha256 == second.sample_sha256
    assert first.selected_records >= 20
    assert first.selected_records % 4 == 0
    selected = [json.loads(line) for line in (tmp_path / "first" / "sample.jsonl").read_text().splitlines()]
    counts = Counter(row["video_id"] for row in selected)
    assert set(counts.values()) == {4}
    assert "Comment" not in first.model_dump_json()


def test_smoke_run_stops_before_evaluation_and_is_non_publishable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "comments.jsonl"
    rows = [
        {"comment_text": f"Useful workflow {index}", "comment_id": str(index), "video_id": f"v{index}"}
        for index in range(30)
    ]
    source.write_text(
        "".join(f"{json.dumps(row)}\n" for row in rows),
    )
    config = _config(tmp_path, run_id="smoke-contract").model_copy(
        update={"source_dataset": source, "expected_records": 30},
    )
    monkeypatch.setattr(
        "src.operations.ml_pipeline.dry_run_pipeline",
        lambda *_args, **_kwargs: type("Allowed", (), {"can_run": True})(),
    )

    report = run_smoke_pipeline(config, PROJECT_ROOT, records=20, executor=_successful_executor)

    assert report.status == DryRunStatus.READY
    assert report.pipeline_status == PipelineStatus.PARTIAL
    assert report.sample.publishable is False
    assert (Path(report.workspace) / "NON_PUBLISHABLE").is_file()
    manifest = json.loads((Path(report.workspace) / "run" / "run-manifest.json").read_text())
    assert len(manifest["stages"]) == 11
    assert manifest["stages"][-1]["stage"] == "reassignment"
    assert not (Path(report.workspace) / "run" / "12-evaluation").exists()
    rendered = render_smoke_run_report(report)
    assert "Full-run decision: allowed" in rendered
    assert "Useful workflow" not in rendered


def test_pipeline_cli_dry_run_and_run_refuse_blocked_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _config(tmp_path, run_id="blocked-cli")
    config.source_dataset.write_text("comment_text,video_id\nprivate text,v1\n")
    config_path = tmp_path / "pipeline.json"
    config_path.write_text(config.model_dump_json())

    assert pipeline_cli.main(["dry-run", str(config_path)]) == 2

    def forbidden_run(*_args: object, **_kwargs: object) -> None:
        msg = "blocked preflight must prevent the real runner"
        raise AssertionError(msg)

    monkeypatch.setattr(pipeline_cli, "run_pipeline", forbidden_run)
    assert pipeline_cli.main(["run", str(config_path)]) == 2
    captured = capsys.readouterr()
    assert "Decision: execution blocked" in captured.out
    assert "private text" not in captured.out
    assert "real run refused" in captured.err
    assert not (config.runs_root / "blocked-cli").exists()


def test_dry_run_safely_reports_wrong_dataset_format(tmp_path: Path) -> None:
    config = _config(tmp_path, run_id="wrong-format")
    config.source_dataset.write_text("comment_text,video_id\nsecret value,v1\n")

    report = dry_run_pipeline(config, PROJECT_ROOT)

    assert report.status == DryRunStatus.BLOCKED
    assert report.dataset is not None
    assert report.dataset.detected_format == "csv"
    assert all(stage.action == DryRunStageAction.BLOCKED for stage in report.stages)
    serialized = report.model_dump_json()
    assert "secret value" not in serialized


def test_dry_run_resume_marks_verified_stages_without_changing_run(tmp_path: Path) -> None:
    config = _config(tmp_path, run_id="resume-dry-run")
    run_dir = tmp_path / "runs" / "resume-dry-run"
    run_pipeline(
        config,
        PROJECT_ROOT,
        stop_after=PipelineStage.CLEANING,
        executor=_successful_executor,
    )
    before = _filesystem_snapshot(run_dir)

    report = dry_run_pipeline(config, PROJECT_ROOT, run_dir=run_dir, resume=True)

    assert _filesystem_snapshot(run_dir) == before
    assert [stage.action for stage in report.stages[:4]] == [DryRunStageAction.SKIP] * 4


def test_dry_run_strictly_validates_configs_without_exposing_values(tmp_path: Path) -> None:
    config = _config(tmp_path, run_id="invalid-config")
    invalid = tmp_path / "invalid-embeddings.json"
    invalid.write_text('{"schema_version":1,"model_name":"private-model","unknown_secret":"do-not-show"}')
    config = config.model_copy(update={"embeddings_config": invalid})

    report = dry_run_pipeline(config, PROJECT_ROOT)

    check = next(item for item in report.checks if item.code == "config_validation.embeddings")
    assert check.status == DryRunStatus.BLOCKED
    assert "private-model" not in report.model_dump_json()
    assert "do-not-show" not in report.model_dump_json()


def test_dry_run_blocks_research_export_and_run_directory_escape(tmp_path: Path) -> None:
    config = _config(tmp_path, run_id="unsafe-export")
    export_config = tmp_path / "unsafe-export.json"
    export_config.write_text(ExportConfig(include_research_text=True).model_dump_json())
    config = config.model_copy(update={"export_config": export_config})
    escaped = tmp_path / "outside-runs" / "unsafe-export"

    report = dry_run_pipeline(config, PROJECT_ROOT, run_dir=escaped)

    blocked = {check.code for check in report.checks if check.status == DryRunStatus.BLOCKED}
    assert "security.export_research_text" in blocked
    assert "run.scope" in blocked


def test_dry_run_detects_invalid_resume_sequence_and_partial_artifacts(tmp_path: Path) -> None:
    config = _config(tmp_path, run_id="invalid-resume")
    run_dir = tmp_path / "runs" / "invalid-resume"
    run_pipeline(
        config,
        PROJECT_ROOT,
        stop_after=PipelineStage.CLEANING,
        executor=_successful_executor,
    )
    manifest_path = run_dir / "run-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["stages"].append(manifest["stages"][-1])
    manifest_path.write_text(json.dumps(manifest))
    partial_dir = run_dir / "05-embeddings"
    partial_dir.mkdir()
    (partial_dir / ".embeddings.partial.npy").write_bytes(b"partial")

    report = dry_run_pipeline(config, PROJECT_ROOT, run_dir=run_dir, resume=True)

    blocked = {check.code for check in report.checks if check.status == DryRunStatus.BLOCKED}
    assert "resume.stage_sequence" in blocked
    assert "resume.partial.embeddings" in blocked


def test_dry_run_restart_marks_force_replacements_and_manual_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, run_id="restart-plan")
    run_dir = tmp_path / "runs" / "restart-plan"
    run_pipeline(
        config,
        PROJECT_ROOT,
        stop_after=PipelineStage.CLEANING,
        executor=_successful_executor,
    )
    monkeypatch.setattr("src.operations.ml_pipeline.importlib.util.find_spec", lambda _name: object())

    report = dry_run_pipeline(
        config,
        PROJECT_ROOT,
        run_dir=run_dir,
        resume=True,
        restart_from=PipelineStage.CLEANING,
    )

    assert [stage.action for stage in report.stages[:3]] == [DryRunStageAction.SKIP] * 3
    cleaning = report.stages[3]
    assert cleaning.action == DryRunStageAction.RESTART
    assert cleaning.requires_force is True
    assert cleaning.replaced_paths
    assert "downstream manual review must be repeated" in cleaning.warnings
    assert any(check.code == "restart.manual_review" for check in report.checks)


def test_dry_run_blocks_export_only_restart_without_final_evaluation(tmp_path: Path) -> None:
    config = _config(tmp_path, run_id="export-restart")
    run_dir = tmp_path / "runs" / "export-restart"
    run_pipeline(
        config,
        PROJECT_ROOT,
        stop_after=PipelineStage.REASSIGNMENT,
        executor=_successful_executor,
    )

    report = dry_run_pipeline(
        config,
        PROJECT_ROOT,
        run_dir=run_dir,
        resume=True,
        restart_from=PipelineStage.EXPORT,
    )

    check = next(item for item in report.checks if item.code == "restart.export_evaluation")
    assert check.status == DryRunStatus.BLOCKED


def test_failed_stage_resume_replaces_failed_record_without_duplicates(tmp_path: Path) -> None:
    config = _config(tmp_path, run_id="failed-resume")
    run_dir = tmp_path / "runs" / "failed-resume"

    def fail_cleaning(command: list[str], log_path: Path, cwd: Path) -> int:
        if Path(command[1]).name == "clean_dataset.py":
            log_path.write_text("safe failure\n")
            return 1
        return _successful_executor(command, log_path, cwd)

    failed = run_pipeline(config, PROJECT_ROOT, executor=fail_cleaning)
    assert failed.status == PipelineStatus.FAILED
    resumed = run_pipeline(
        config,
        PROJECT_ROOT,
        run_dir=run_dir,
        resume=True,
        stop_after=PipelineStage.CLEANING,
        executor=_successful_executor,
    )
    assert resumed.status == PipelineStatus.PARTIAL
    assert [record.stage for record in resumed.stages] == list(PipelineStage)[:4]
    assert resumed.stages[-1].command[-1] == "--force"


def test_pipeline_stops_at_manual_review_gate_and_records_safe_logs(tmp_path: Path) -> None:
    config = _config(tmp_path, run_id="review-run")
    result = run_pipeline(config, PROJECT_ROOT, executor=_preliminary_evaluation_executor)
    assert result.status == PipelineStatus.AWAITING_REVIEW
    assert result.stages[-1].stage == PipelineStage.EVALUATION
    assert "manual annotations" in result.message
    assert not (tmp_path / "runs" / "review-run" / "13-export" / "export-manifest.json").exists()
    persisted = json.loads((tmp_path / "runs" / "review-run" / "run-manifest.json").read_text())
    assert persisted["status"] == "awaiting_review"


def _passing_export(tmp_path: Path) -> Path:
    annotations = tmp_path / "annotations.jsonl"
    annotations.parent.mkdir(parents=True, exist_ok=True)
    annotations.write_text(
        '{"record_index":0,"topic_matches":true,"topic_clear":true,"business_relevant":true}\n'
        '{"record_index":5,"topic_matches":true,"topic_clear":true,"business_relevant":true,'
        '"reassignment_correct":true}\n',
    )
    _, evaluation_dir = _run_evaluation(tmp_path, annotations=annotations)
    export_dir = tmp_path / "export"
    export_topic_results(
        tmp_path / "corpus-manifest.json",
        tmp_path / "clustering-manifest.json",
        tmp_path / "topics" / "topic-representation-manifest.json",
        tmp_path / "reassignment" / "outlier-reassignment-manifest.json",
        evaluation_dir / "evaluation-manifest.json",
        export_dir,
        config=ExportConfig(require_final_evaluation=True),
    )
    return export_dir


def test_publication_and_rollback_use_atomic_validated_links(tmp_path: Path) -> None:
    first = _passing_export(tmp_path / "first")
    second = _passing_export(tmp_path / "second")
    publish_root = tmp_path / "published"

    current = publish_snapshot(first, publish_root, "first")
    assert current.resolve() == (first / "export-manifest.json").resolve()
    current = publish_snapshot(second, publish_root, "second")
    assert current.resolve() == (second / "export-manifest.json").resolve()
    rolled_back = rollback_snapshot(publish_root)
    assert rolled_back.resolve() == (first / "export-manifest.json").resolve()
