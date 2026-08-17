import json
from pathlib import Path

import pytest

from scripts.clean_dataset import main
from src.ml.cleaning_dataset import (
    CleaningDecisionRecord,
    DatasetCleaningManifest,
    clean_development_dataset,
)
from src.ml.config import CleaningConfig
from src.ml.models import CleaningReason
from src.ml.schemas import CleanedTextUnit, TextKind
from src.ml.splitting import SplitConfig, SplitName, split_comments_jsonl


def _build_split(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "raw.jsonl"
    rows = [
        {
            "comment_id": "one",
            "comment_text": "Main CRM integration pain",
            "video_id": "cleaning-video",
            "comment_replies": [
                {"text": "Reply inventory automation pain", "author_display_name": "Private Reply Author"},
                {"text": "Main CRM integration pain"},
                {"text": "Спасибо"},
            ],
        },
        {
            "comment_id": "two",
            "comment_text": " main crm integration pain!!! ",
            "video_id": "cleaning-video",
        },
        {
            "comment_id": "three",
            "comment_text": "https://example.com/private",
            "video_id": "cleaning-video",
        },
    ]
    source.write_text("".join(f"{json.dumps(row, ensure_ascii=False)}\n" for row in rows), encoding="utf-8")
    split_dir = tmp_path / "splits"
    split_manifest = split_comments_jsonl(
        source,
        split_dir,
        config=SplitConfig(development_ratio=0.98, validation_ratio=0.01, test_ratio=0.01),
    )
    assert split_manifest.stats.written_records[SplitName.DEVELOPMENT] == 3
    return split_dir / "development.jsonl", split_dir / "split-manifest.json"


def test_cleaning_flattens_replies_deduplicates_and_audits_decisions(tmp_path: Path) -> None:
    source, split_manifest = _build_split(tmp_path)
    output_dir = tmp_path / "cleaning"
    config = CleaningConfig(min_length=1, detect_language=False)

    manifest = clean_development_dataset(source, split_manifest, output_dir, config=config)

    cleaned = [
        CleanedTextUnit.model_validate_json(line)
        for line in (output_dir / "development-clean.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    decisions = [
        CleaningDecisionRecord.model_validate_json(line)
        for line in (output_dir / "cleaning-decisions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert manifest.stats.input_rows == 3
    assert manifest.stats.input_text_units == 6
    assert manifest.stats.input_replies == 3
    assert manifest.stats.output_text_units == 3
    assert {unit.text_kind for unit in cleaned} == {TextKind.COMMENT, TextKind.REPLY}
    assert next(unit for unit in cleaned if unit.text_kind == TextKind.COMMENT).duplicate_count == 2
    same_text_roles = {unit.text_kind for unit in cleaned if unit.clean_text.casefold() == "main crm integration pain"}
    assert same_text_roles == {TextKind.COMMENT, TextKind.REPLY}
    assert sum(decision.reason == CleaningReason.EXACT_DUPLICATE for decision in decisions) == 1
    assert sum(decision.reason == CleaningReason.ACKNOWLEDGEMENT for decision in decisions) == 1
    assert sum(decision.reason == CleaningReason.URL_ONLY for decision in decisions) == 1
    assert manifest.output_sha256
    assert DatasetCleaningManifest.model_validate_json(
        (output_dir / "cleaning-manifest.json").read_text(encoding="utf-8"),
    ) == manifest


def test_cleaning_reports_and_decisions_do_not_leak_values(tmp_path: Path) -> None:
    source, split_manifest = _build_split(tmp_path)
    output_dir = tmp_path / "cleaning"

    clean_development_dataset(
        source,
        split_manifest,
        output_dir,
        config=CleaningConfig(min_length=1, detect_language=False),
    )

    safe_artifacts = [
        output_dir / "cleaning-report.md",
        output_dir / "cleaning-decisions.jsonl",
        output_dir / "aggregate-tables" / "removed-by-reason.csv",
        output_dir / "aggregate-tables" / "detected-languages.csv",
    ]
    combined = "".join(path.read_text(encoding="utf-8") for path in safe_artifacts)
    assert "Private Reply Author" not in combined
    assert "inventory automation" not in combined
    assert "example.com/private" not in combined
    assert "cleaning-video" not in combined


def test_cleaning_rejects_tampering_and_existing_outputs(tmp_path: Path) -> None:
    source, split_manifest = _build_split(tmp_path)
    output_dir = tmp_path / "cleaning"
    config = CleaningConfig(min_length=1, detect_language=False)
    clean_development_dataset(source, split_manifest, output_dir, config=config)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        clean_development_dataset(source, split_manifest, output_dir, config=config)

    source.write_text('{"comment_text":"tampered"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        clean_development_dataset(source, split_manifest, tmp_path / "tampered", config=config)


def test_cleaning_cli_uses_json_config(tmp_path: Path) -> None:
    source, split_manifest = _build_split(tmp_path)
    config_path = tmp_path / "cleaning.json"
    config_path.write_text(
        CleaningConfig(min_length=1, detect_language=False).model_dump_json(),
        encoding="utf-8",
    )
    output_dir = tmp_path / "cleaning"

    exit_code = main(
        [
            str(source),
            "--manifest",
            str(split_manifest),
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert exit_code == 0
    assert (output_dir / "development-clean.jsonl").is_file()
    assert (output_dir / "cleaning-manifest.json").is_file()
