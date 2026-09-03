import json
import time
from pathlib import Path

import pytest

from src.operations import (
    ProgressEvent,
    ProgressJournal,
    ProgressStatus,
    make_progress_event,
    render_progress_event,
)
from src.operations.ml_pipeline import PipelineStage, _execute_with_heartbeat


def test_progress_event_derives_percentage_without_private_fields() -> None:
    event = make_progress_event(
        run_id="run-1",
        stage="embeddings",
        stage_number=5,
        status=ProgressStatus.PROGRESS,
        completed=25,
        total=100,
        message="batch completed",
    )

    assert event.progress == 0.25
    assert "25/100 (25.00%)" in render_progress_event(event)
    assert "text" not in ProgressEvent.model_fields
    assert "author" not in ProgressEvent.model_fields


def test_progress_event_rejects_impossible_counts() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        make_progress_event(
            run_id="run-1",
            stage="cleaning",
            stage_number=4,
            status=ProgressStatus.PROGRESS,
            completed=2,
            total=1,
        )


def test_progress_journal_appends_events_and_updates_current_status(tmp_path: Path) -> None:
    journal = ProgressJournal(tmp_path)
    started = make_progress_event(
        run_id="run-1",
        stage="pipeline",
        stage_number=0,
        status=ProgressStatus.STARTED,
    )
    completed = started.model_copy(update={"status": ProgressStatus.COMPLETED})

    journal(started)
    journal(completed)

    lines = journal.events_path.read_text(encoding="utf-8").splitlines()
    current = ProgressEvent.model_validate_json(journal.current_path.read_text(encoding="utf-8"))
    assert len(lines) == 2
    assert [json.loads(line)["status"] for line in lines] == ["started", "completed"]
    assert current.status == ProgressStatus.COMPLETED
    assert not list(journal.directory.glob("*.tmp"))


def test_stage_aware_heartbeat_stops_before_executor_returns(tmp_path: Path) -> None:
    events: list[ProgressEvent] = []

    def executor(_command: list[str], _log_path: Path, _cwd: Path) -> int:
        time.sleep(0.04)
        return 7

    result = _execute_with_heartbeat(
        executor,
        ["python", "stage.py"],
        tmp_path / "stage.log",
        tmp_path,
        run_id="run-1",
        stage=PipelineStage.REDUCTION,
        progress_callback=events.append,
        heartbeat_interval=0.01,
    )
    observed = len(events)
    time.sleep(0.02)

    assert result == 7
    assert observed >= 2
    assert len(events) == observed
    assert {event.run_id for event in events} == {"run-1"}
    assert {event.stage for event in events} == {"reduction"}
    assert {event.status for event in events} == {ProgressStatus.HEARTBEAT}


def test_heartbeat_callback_failure_does_not_change_executor_result(tmp_path: Path) -> None:
    def broken_callback(_event: ProgressEvent) -> None:
        message = "display unavailable"
        raise RuntimeError(message)

    def executor(_command: list[str], _log_path: Path, _cwd: Path) -> int:
        time.sleep(0.02)
        return 0

    assert (
        _execute_with_heartbeat(
            executor,
            ["python", "stage.py"],
            tmp_path / "stage.log",
            tmp_path,
            run_id="run-1",
            stage=PipelineStage.CLUSTERING,
            progress_callback=broken_callback,
            heartbeat_interval=0.005,
        )
        == 0
    )
