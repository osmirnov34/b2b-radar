import json
import sys
from pathlib import Path

import pytest

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
    dry_run_pipeline,
    publish_snapshot,
    rollback_snapshot,
    run_pipeline,
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
