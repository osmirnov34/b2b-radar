import json
from pathlib import Path

import pytest

from scripts.run_eda import main
from src.analysis.eda import EDAConfig, profile_development_dataset, summarize_numbers, write_eda_reports
from src.analysis.splitting import SplitConfig, SplitName, split_comments_jsonl


def _make_development_split(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "comments.jsonl"
    rows = [
        {
            "comment_id": str(index),
            "comment_text": text,
            "comment_author": author,
            "video_id": "video-eda",
            "video_channel": "private-channel",
            "search_query": "private-query",
            "comment_published_at": "2026-01-10T10:00:00Z",
        }
        for index, (text, author) in enumerate(
            [
                ("Спасибо", "alice"),
                ("Не могу настроить интеграцию с CRM", "bob"),
                ("Не могу настроить интеграцию с CRM", "bob"),
                ("VISIT HTTPS://EXAMPLE.COM NOW PLEASE", ""),
            ],
        )
    ]
    source.write_text("".join(f"{json.dumps(row, ensure_ascii=False)}\n" for row in rows), encoding="utf-8")
    output = tmp_path / "splits"
    manifest = split_comments_jsonl(
        source,
        output,
        config=SplitConfig(development_ratio=0.98, validation_ratio=0.01, test_ratio=0.01),
    )
    assert manifest.stats.written_records[SplitName.DEVELOPMENT] == len(rows)
    return output / "development.jsonl", output / "split-manifest.json"


def test_numeric_summary_handles_empty_and_quantiles() -> None:
    assert summarize_numbers([]) is None
    summary = summarize_numbers([1, 2, 3, 4, 100])
    assert summary is not None
    assert summary.minimum == 1
    assert summary.p50 == 3
    assert summary.p95 == 100
    assert summary.maximum == 100


def test_profile_development_computes_safe_aggregates(tmp_path: Path) -> None:
    development, manifest = _make_development_split(tmp_path)

    profile = profile_development_dataset(
        development,
        manifest,
        config=EDAConfig(language_sample_size=0, top_groups=5),
    )

    serialized = profile.model_dump_json()
    assert profile.records == 4
    assert profile.unique_texts == 3
    assert profile.duplicate_texts == 1
    assert profile.duplicate_groups == 1
    assert profile.unique_authors == 2
    assert profile.noise_categories["acknowledgement"] == 1
    assert profile.noise_categories["contains_url"] == 1
    assert profile.monthly_records == {"2026-01": 4}
    assert "private-channel" not in serialized
    assert "private-query" not in serialized
    assert "Не могу настроить" not in serialized


def test_profile_rejects_non_development_checksum(tmp_path: Path) -> None:
    development, manifest = _make_development_split(tmp_path)
    development.write_text('{"comment_text":"tampered"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="checksum"):
        profile_development_dataset(development, manifest, config=EDAConfig(language_sample_size=0))


def test_reports_are_complete_and_do_not_contain_raw_values(tmp_path: Path) -> None:
    development, manifest = _make_development_split(tmp_path)
    profile = profile_development_dataset(development, manifest, config=EDAConfig(language_sample_size=0))

    paths = write_eda_reports(profile, tmp_path / "reports")

    assert len(paths) == 12
    combined = "".join(path.read_text(encoding="utf-8") for path in paths)
    assert "private-channel" not in combined
    assert "private-query" not in combined
    assert "Не могу настроить" not in combined
    assert "<svg" in (tmp_path / "reports" / "figures" / "character-lengths.svg").read_text(encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_eda_reports(profile, tmp_path / "reports")


def test_eda_cli_profiles_only_manifest_development(tmp_path: Path) -> None:
    development, manifest = _make_development_split(tmp_path)
    report_dir = tmp_path / "reports"

    exit_code = main(
        [
            str(development),
            "--manifest",
            str(manifest),
            "--report-dir",
            str(report_dir),
            "--language-sample-size",
            "0",
        ],
    )

    assert exit_code == 0
    assert (report_dir / "development-profile.json").is_file()
