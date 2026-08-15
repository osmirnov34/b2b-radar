import json
from pathlib import Path

import pytest

from scripts.inspect_dataset import main
from src.analysis import (
    DatasetInspection,
    SampleMetadata,
    create_research_sample,
    inspect_comments_jsonl,
    write_inspection_reports,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_inspection_reports_are_written_without_comment_values(tmp_path: Path) -> None:
    source = tmp_path / "broken.jsonl"
    source.write_text('{"text":"private comment"}\n', encoding="utf-8")
    report = inspect_comments_jsonl(source, max_error_rate=1)

    json_path, markdown_path, errors_path = write_inspection_reports(report, tmp_path / "reports")

    restored = DatasetInspection.model_validate_json(json_path.read_text(encoding="utf-8"))
    combined = markdown_path.read_text(encoding="utf-8") + errors_path.read_text(encoding="utf-8")
    assert restored.sha256 == report.sha256
    assert "private comment" not in combined


def test_research_sample_is_reproducible_and_limits_each_video(tmp_path: Path) -> None:
    source = tmp_path / "comments.jsonl"
    rows = [
        {"comment_id": str(index), "comment_text": f"Comment {index}", "video_id": "dominant"}
        for index in range(10)
    ] + [
        {"comment_id": "other", "comment_text": "Other comment", "video_id": "other"},
    ]
    source.write_text("".join(f"{json.dumps(row)}\n" for row in rows), encoding="utf-8")
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"

    first_meta = create_research_sample(source, first, sample_size=10, seed=7, max_records_per_video=2)
    second_meta = create_research_sample(source, second, sample_size=10, seed=7, max_records_per_video=2)
    sampled = [json.loads(line) for line in first.read_text(encoding="utf-8").splitlines()]

    assert first.read_bytes() == second.read_bytes()
    assert first_meta.output_sha256 == second_meta.output_sha256
    assert first_meta.written_records == 3
    assert sum(row["video_id"] == "dominant" for row in sampled) == 2
    metadata = SampleMetadata.model_validate_json(first.with_suffix(".jsonl.meta.json").read_text(encoding="utf-8"))
    assert metadata.source_sha256 == first_meta.source_sha256


def test_dataset_cli_writes_reports_and_sample(tmp_path: Path) -> None:
    report_dir = tmp_path / "reports"
    sample_path = tmp_path / "sample.jsonl"

    exit_code = main(
        [
            str(FIXTURES_DIR / "comments.jsonl"),
            "--report-dir",
            str(report_dir),
            "--expected-records",
            "2",
            "--sample-size",
            "1",
            "--sample-output",
            str(sample_path),
        ],
    )

    assert exit_code == 0
    assert (report_dir / "dataset-profile.json").is_file()
    assert sample_path.is_file()


def test_dataset_cli_reports_detected_format_on_failure(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    html = tmp_path / "comments.jsonl"
    html.write_text("<!doctype html><title>Google Drive</title>", encoding="utf-8")

    exit_code = main([str(html), "--report-dir", str(tmp_path / "reports")])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Detected format: html" in captured.out
    assert "expected jsonl, detected html" in captured.err
