import json
from pathlib import Path

import pytest

from src.analysis import DatasetFormat, detect_dataset_format, inspect_comments_jsonl

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize(
    ("filename", "content", "expected"),
    [
        ("data.jsonl", b'{"comment_text":"one"}\n{"comment_text":"two"}\n', DatasetFormat.JSONL),
        ("data.json", b'{"comment_text":"one"}', DatasetFormat.JSON_OBJECT),
        ("data.json", b'[{"comment_text":"one"}]', DatasetFormat.JSON_ARRAY),
        ("data.csv", b"comment_text,video_id\none,v1\n", DatasetFormat.CSV),
        ("data.tsv", b"comment_text\tvideo_id\none\tv1\n", DatasetFormat.TSV),
        ("data.jsonl", b"<!DOCTYPE html><html><body>Google Drive</body></html>", DatasetFormat.HTML),
        ("data.jsonl", b"PK\x03\x04archive", DatasetFormat.ZIP),
        ("data.jsonl", b"\x1f\x8barchive", DatasetFormat.GZIP),
        ("data.jsonl", b"\xff\xfe\x00\x01", DatasetFormat.BINARY),
        ("data.jsonl", b"", DatasetFormat.EMPTY),
    ],
)
def test_detect_dataset_format(tmp_path: Path, filename: str, content: bytes, expected: DatasetFormat) -> None:
    path = tmp_path / filename
    path.write_bytes(content)

    result = detect_dataset_format(path)

    assert result.detected == expected
    assert result.matches is (expected == DatasetFormat.JSONL)


def test_inspection_profiles_valid_fixture_without_exposing_text() -> None:
    report = inspect_comments_jsonl(FIXTURES_DIR / "comments.jsonl", expected_records=2)

    assert report.is_usable
    assert report.contract_valid == 2
    assert report.non_empty_text == 2
    assert report.unique_authors == 2
    assert report.unique_videos == 2
    assert report.record_count_matches_expectation is True
    assert report.sha256
    assert report.errors == []


def test_inspection_reports_schema_and_json_errors_safely(tmp_path: Path) -> None:
    path = tmp_path / "broken.jsonl"
    path.write_text(
        '{"text":"wrong field","author":"Secret author"}\nnot-json\n{"comment_text":"Valid"}\n',
        encoding="utf-8",
    )

    report = inspect_comments_jsonl(path, max_error_rate=1)

    assert report.contract_valid == 1
    assert report.contract_invalid == 1
    assert report.json_invalid == 1
    assert {error.error_type for error in report.errors} == {"schema", "json_decode"}
    serialized_errors = json.dumps([error.model_dump(mode="json") for error in report.errors])
    assert "Secret author" not in serialized_errors
    assert "wrong field" not in serialized_errors
    assert report.errors[0].fields == ["author", "text"]


def test_inspection_marks_format_mismatch_as_critical(tmp_path: Path) -> None:
    path = tmp_path / "comments.jsonl"
    path.write_text("<!doctype html><title>Drive</title>", encoding="utf-8")

    report = inspect_comments_jsonl(path)

    assert not report.is_usable
    assert report.format.detected == DatasetFormat.HTML
    assert "expected jsonl, detected html" in report.critical_errors


def test_inspection_counts_exact_and_id_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "comments.jsonl"
    rows = [
        {"comment_id": "1", "comment_text": "CRM для бизнеса", "video_id": "v1"},
        {"comment_id": "2", "comment_text": " crm  ДЛЯ бизнеса ", "video_id": "v1"},
        {"comment_id": "1", "comment_text": "Другой текст", "video_id": "v2"},
    ]
    path.write_text("".join(f"{json.dumps(row, ensure_ascii=False)}\n" for row in rows), encoding="utf-8")

    report = inspect_comments_jsonl(path)

    assert report.duplicate_texts == 1
    assert report.duplicate_text_groups == 1
    assert report.largest_duplicate_text_group == 2
    assert report.duplicate_comment_ids == 1
    assert report.conflicting_comment_ids == 1
