import hashlib
import json
from pathlib import Path

import pytest

from scripts.split_dataset import main
from src.analysis.splitting import (
    DatasetSplitManifest,
    SplitConfig,
    SplitName,
    is_informative_leakage_text,
    split_comments_jsonl,
)


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text("".join(f"{json.dumps(row, ensure_ascii=False)}\n" for row in rows), encoding="utf-8")


def _read_rows(path: Path) -> list[dict[str, str]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_split_keeps_each_video_in_one_partition_and_is_reproducible(tmp_path: Path) -> None:
    source = tmp_path / "comments.jsonl"
    rows = [
        {
            "comment_id": f"{video}-{index}",
            "comment_text": f"Useful long comment {video} number {index}",
            "video_id": video,
        }
        for video in ("a", "b", "c", "d", "e", "f", "g", "h", "i", "j")
        for index in range(3)
    ]
    _write_rows(source, rows)

    first = split_comments_jsonl(source, tmp_path / "first", config=SplitConfig(seed=7))
    second = split_comments_jsonl(source, tmp_path / "second", config=SplitConfig(seed=7))

    locations: dict[str, set[SplitName]] = {}
    for split in SplitName:
        assert (tmp_path / "first" / f"{split.value}.jsonl").read_bytes() == (
            tmp_path / "second" / f"{split.value}.jsonl"
        ).read_bytes()
        for row in _read_rows(tmp_path / "first" / f"{split.value}.jsonl"):
            locations.setdefault(row["video_id"], set()).add(split)
    assert all(len(splits) == 1 for splits in locations.values())
    assert first.output_sha256 == second.output_sha256


def test_content_leaks_are_removed_but_short_noise_overlaps_remain(tmp_path: Path) -> None:
    config = SplitConfig(seed=42)
    source = tmp_path / "comments.jsonl"
    unique = "This is a sufficiently detailed repeated business problem"
    rows: list[dict[str, str]] = []
    # Find one group in every split without coupling the test to hard-coded hash outcomes.
    selected: dict[SplitName, str] = {}
    for index in range(1000):
        video_id = f"video-{index}"
        digest = hashlib.sha256(f"{config.seed}\x1fvideo_id:{video_id}".encode()).digest()
        fraction = int.from_bytes(digest[:8]) / 2**64
        if fraction < config.development_ratio:
            split = SplitName.DEVELOPMENT
        elif fraction < config.development_ratio + config.validation_ratio:
            split = SplitName.VALIDATION
        else:
            split = SplitName.TEST
        selected.setdefault(split, video_id)
        if len(selected) == 3:
            break
    assert len(selected) == 3
    for split, video_id in selected.items():
        rows.extend(
            [
                {"comment_id": f"{split}-content", "comment_text": unique, "video_id": video_id},
                {"comment_id": f"{split}-noise", "comment_text": "Спасибо", "video_id": video_id},
            ],
        )
    _write_rows(source, rows)

    manifest = split_comments_jsonl(source, tmp_path / "result", config=config)

    development = _read_rows(tmp_path / "result" / "development.jsonl")
    validation = _read_rows(tmp_path / "result" / "validation.jsonl")
    test = _read_rows(tmp_path / "result" / "test.jsonl")
    assert [row["comment_text"] for row in development].count(unique) == 1
    assert all(row["comment_text"] != unique for row in validation + test)
    assert sum(row["comment_text"] == "Спасибо" for row in development + validation + test) == 3
    assert manifest.stats.removed_content_leaks[SplitName.VALIDATION] == 1
    assert manifest.stats.removed_content_leaks[SplitName.TEST] == 1
    assert manifest.stats.ignored_noise_overlaps == 1


def test_split_refuses_overwrite_and_rejects_invalid_input(tmp_path: Path) -> None:
    source = tmp_path / "comments.jsonl"
    _write_rows(source, [{"comment_text": "A valid and sufficiently detailed comment", "video_id": "video"}])
    output = tmp_path / "output"
    split_comments_jsonl(source, output)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        split_comments_jsonl(source, output)

    source.write_text("not-json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        split_comments_jsonl(source, tmp_path / "invalid")


def test_informative_leakage_thresholds_ignore_generic_phrases() -> None:
    config = SplitConfig(min_informative_chars=20, min_informative_tokens=4)

    assert not is_informative_leakage_text("Спасибо", config)
    assert not is_informative_leakage_text("Очень круто!", config)
    assert is_informative_leakage_text("Не могу настроить интеграцию с CRM", config)


def test_split_cli_writes_manifest_and_privacy_safe_report(tmp_path: Path) -> None:
    source = tmp_path / "comments.jsonl"
    private_text = "Private detailed customer problem with integration setup"
    _write_rows(source, [{"comment_text": private_text, "video_id": "private-video-id"}])
    output = tmp_path / "splits"

    exit_code = main([str(source), "--output-dir", str(output)])

    assert exit_code == 0
    manifest_text = (output / "split-manifest.json").read_text(encoding="utf-8")
    report_text = (output / "split-report.md").read_text(encoding="utf-8")
    manifest = DatasetSplitManifest.model_validate_json(manifest_text)
    assert manifest.stats.input_records == 1
    assert private_text not in manifest_text + report_text
    assert "private-video-id" not in manifest_text + report_text


def test_split_cli_returns_error_for_existing_outputs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "comments.jsonl"
    _write_rows(source, [{"comment_text": "Detailed useful comment for splitting", "video_id": "video"}])
    output = tmp_path / "splits"
    assert main([str(source), "--output-dir", str(output)]) == 0

    assert main([str(source), "--output-dir", str(output)]) == 1
    assert "refusing to overwrite" in capsys.readouterr().err
