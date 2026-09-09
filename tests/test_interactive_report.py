import hashlib
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from pydantic import BaseModel

from src.ml import interactive_report
from src.ml.interactive_report import InteractiveReportConfig, write_interactive_report
from src.ml.reporting import AnalysisArtifacts


class _Row(BaseModel):
    topic_id: int
    name: str = "Topic <script>alert(1)</script>"
    records: int = 8
    corpus_share: float = 0.8
    mean_probability: float = 0.75
    unique_videos: int = 3
    keywords: list[str] = ["delivery", "delay"]
    comments: int = 6
    replies: int = 2
    reassigned_outliers: int = 1
    signal_share: float = 0.25
    status: str = "problem_candidate"
    topic_status: str = "distributed"
    latest_month: str = "2026-08"
    relative_change: float = 0.2
    priority_score: float = 72.0


class _Example(BaseModel):
    topic_id: int = 0
    text: str = "Private </script> comment"
    centroid_similarity: float = 0.93
    video_url: str = "https://www.youtube.com/watch?v=abc123"


class _Lineage(BaseModel):
    metrics: list[dict[str, object]] = [
        {
            "label": "Valid parent comments",
            "records": 10,
            "scope": "all",
            "description": "Validated input parents",
        },
    ]
    checks: list[dict[str, object]] = []
    cleaning_removed_by_reason: dict[str, int] = {}
    semantic_duplicates_removed: int = 0


def _artifacts(tmp_path: Path) -> AnalysisArtifacts:
    return cast(
        "AnalysisArtifacts",
        SimpleNamespace(
            run_dir=tmp_path / "ml-runs/run-1",
            run_id="run-1",
            pipeline_manifest_sha256="a" * 64,
            summary=SimpleNamespace(
                model_dump=lambda **_kwargs: {
                    "records": 10,
                    "topics": 1,
                    "outliers": 2,
                    "outlier_share": 0.2,
                    "mean_confidence": 0.7,
                    "largest_topic_share": 0.8,
                    "pipeline_status": "awaiting_review",
                    "warnings": ["manual review pending"],
                },
            ),
        ),
    )


def _stub_report_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    row = _Row(topic_id=0)
    monkeypatch.setattr(interactive_report, "build_data_lineage", lambda _artifacts: _Lineage())
    monkeypatch.setattr(interactive_report, "build_cluster_cards", lambda _artifacts: [row])
    monkeypatch.setattr(interactive_report, "topic_summary_rows", lambda _artifacts: [row])
    monkeypatch.setattr(interactive_report, "assess_topic_problem_signals", lambda *_args, **_kwargs: [row])
    monkeypatch.setattr(interactive_report, "rank_problem_topics", lambda *_args, **_kwargs: [row])
    monkeypatch.setattr(interactive_report, "assess_video_concentration", lambda *_args, **_kwargs: [row])
    monkeypatch.setattr(interactive_report, "analyze_topic_temporal_trends", lambda *_args, **_kwargs: [row])


def test_interactive_report_is_portable_checksum_bound_and_private_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_report_inputs(monkeypatch)
    monkeypatch.setattr(
        interactive_report,
        "representative_comments",
        lambda *_args, **_kwargs: pytest.fail("private examples must not be loaded"),
    )
    output = tmp_path / "reports/run-1"

    manifest = write_interactive_report(
        _artifacts(tmp_path),
        output,
        analyzed_at=datetime(2026, 9, 1, tzinfo=UTC),
    )
    html = (output / "interactive-report.html").read_text(encoding="utf-8")

    assert manifest.classification == "aggregate_internal"
    assert manifest.private_text_included is False
    assert manifest.examples == 0
    assert manifest.report_sha256 == hashlib.sha256(html.encode()).hexdigest()
    assert "https://cdn" not in html
    assert "Поиск по названию" in html
    assert "Valid parent comments" in html
    assert "<script>alert(1)</script>" not in html
    assert "\\u003cscript>alert(1)\\u003c/script>" in html


def test_interactive_report_requires_explicit_private_examples_and_marks_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_report_inputs(monkeypatch)
    monkeypatch.setattr(interactive_report, "representative_comments", lambda *_args, **_kwargs: [_Example()])
    output = tmp_path / "restricted/run-1"

    manifest = write_interactive_report(
        _artifacts(tmp_path),
        output,
        config=InteractiveReportConfig(include_private_examples=True),
        analyzed_at=datetime(2026, 9, 1, tzinfo=UTC),
    )
    html = (output / "interactive-report.html").read_text(encoding="utf-8")

    assert manifest.classification == "restricted"
    assert manifest.private_text_included is True
    assert manifest.examples == 1
    assert "Private </script> comment" not in html
    assert "Private \\u003c/script> comment" in html


def test_interactive_report_refuses_unsafe_time_location_and_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_report_inputs(monkeypatch)
    monkeypatch.setattr(interactive_report, "representative_comments", lambda *_args, **_kwargs: [])
    artifacts = _artifacts(tmp_path)

    with pytest.raises(ValueError, match="timezone-aware"):
        write_interactive_report(
            artifacts,
            tmp_path / "reports",
            analyzed_at=datetime(2026, 9, 1),  # noqa: DTZ001 - deliberately invalid input
        )
    with pytest.raises(ValueError, match="outside the immutable run"):
        write_interactive_report(
            artifacts,
            artifacts.run_dir / "report",
            analyzed_at=datetime(2026, 9, 1, tzinfo=UTC),
        )

    output = tmp_path / "reports"
    write_interactive_report(artifacts, output, analyzed_at=datetime(2026, 9, 1, tzinfo=UTC))
    with pytest.raises(FileExistsError, match="already exists"):
        write_interactive_report(artifacts, output, analyzed_at=datetime(2026, 9, 1, tzinfo=UTC))


def test_interactive_report_rejects_symbolic_link_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_report_inputs(monkeypatch)
    real_output = tmp_path / "real-output"
    real_output.mkdir()
    linked_output = tmp_path / "linked-output"
    linked_output.symlink_to(real_output, target_is_directory=True)

    with pytest.raises(ValueError, match="symbolic link"):
        write_interactive_report(
            _artifacts(tmp_path),
            linked_output,
            analyzed_at=datetime(2026, 9, 1, tzinfo=UTC),
        )
