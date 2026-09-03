"""Versioned, privacy-safe progress events for long-running pipeline orchestration."""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from pathlib import Path


class ProgressStatus(StrEnum):
    STARTED = "started"
    PROGRESS = "progress"
    CHECKPOINT_SAVED = "checkpoint_saved"
    HEARTBEAT = "heartbeat"
    COMPLETED = "completed"
    WARNING = "warning"
    FAILED = "failed"
    SKIPPED = "skipped"
    RESUMED = "resumed"


class ProgressEvent(BaseModel):
    """One aggregate event; record text and provenance are deliberately unsupported."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    run_id: str = Field(min_length=1)
    stage: str
    stage_number: int = Field(ge=0)
    status: ProgressStatus
    completed: int | None = Field(default=None, ge=0)
    total: int | None = Field(default=None, ge=0)
    progress: float | None = Field(default=None, ge=0, le=1)
    elapsed_seconds: float | None = Field(default=None, ge=0)
    message: str = ""
    metrics: dict[str, int | float | str | bool] = Field(default_factory=dict)


class ProgressCallback(Protocol):
    def __call__(self, event: ProgressEvent) -> None: ...


def make_progress_event(
    *,
    run_id: str,
    stage: str,
    stage_number: int,
    status: ProgressStatus,
    completed: int | None = None,
    total: int | None = None,
    elapsed_seconds: float | None = None,
    message: str = "",
    metrics: dict[str, int | float | str | bool] | None = None,
) -> ProgressEvent:
    """Build an event and derive progress only when a meaningful total exists."""
    if completed is not None and total is not None and completed > total:
        msg = "completed progress cannot exceed total"
        raise ValueError(msg)
    progress = completed / total if completed is not None and total not in {None, 0} else None
    return ProgressEvent(
        run_id=run_id,
        stage=stage,
        stage_number=stage_number,
        status=status,
        completed=completed,
        total=total,
        progress=progress,
        elapsed_seconds=elapsed_seconds,
        message=message,
        metrics=metrics or {},
    )


def render_progress_event(event: ProgressEvent) -> str:
    """Render one compact line suitable for a Colab output cell."""
    timestamp = event.timestamp.astimezone(UTC).strftime("%H:%M:%S")
    position = f" {event.completed}/{event.total} ({event.progress:.2%})" if event.progress is not None else ""
    elapsed = f" elapsed={event.elapsed_seconds:.1f}s" if event.elapsed_seconds is not None else ""
    message = f" — {event.message}" if event.message else ""
    return (
        f"[{timestamp}] [{event.stage_number:02d}] {event.stage.upper()} "
        f"{event.status.value.upper()}{position}{elapsed}{message}"
    )


class ProgressJournal:
    """Append events and atomically publish the current aggregate status."""

    def __init__(self, run_dir: Path, *, echo: bool = False) -> None:
        self.directory = run_dir / "observability"
        self.events_path = self.directory / "progress-events.jsonl"
        self.current_path = self.directory / "current-status.json"
        self.echo = echo

    def __call__(self, event: ProgressEvent) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8") as target:
            target.write(f"{event.model_dump_json()}\n")
            target.flush()
            os.fsync(target.fileno())
        temporary = self.current_path.with_name(f".{self.current_path.name}.tmp")
        temporary.write_text(f"{event.model_dump_json(indent=2)}\n", encoding="utf-8")
        temporary.replace(self.current_path)
        if self.echo:
            sys.stdout.write(f"{render_progress_event(event)}\n")
            sys.stdout.flush()


class CompositeProgressCallback:
    """Fan out progress without coupling the pipeline to a specific user interface."""

    def __init__(self, *callbacks: ProgressCallback) -> None:
        self.callbacks = callbacks

    def __call__(self, event: ProgressEvent) -> None:
        for callback in self.callbacks:
            callback(event)
