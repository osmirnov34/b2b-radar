"""Reproducible cleaning of checksum-verified development text units."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from csv import writer
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, BinaryIO

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.ml.cleaning import classify_prepared_text, prepare_text
from src.ml.config import CleaningConfig
from src.ml.models import CleaningReason
from src.ml.schemas import CleanedTextUnit, ExportedComment, TextKind, TextUnit, exported_comment_to_text_units
from src.ml.splitting import DatasetSplitManifest, SplitName
from src.text_processing.language import detect_language

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

_LOW_RETENTION_WARNING = 0.20


class _CleaningModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CleaningDecisionRecord(_CleaningModel):
    record_index: int = Field(ge=0)
    record_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    text_kind: TextKind
    kept: bool
    reason: CleaningReason | None = None
    detected_language: str
    representative_index: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_decision(self) -> CleaningDecisionRecord:
        if self.kept and self.reason is not None:
            msg = "kept records cannot have a removal reason"
            raise ValueError(msg)
        if not self.kept and self.reason is None:
            msg = "removed records require a reason"
            raise ValueError(msg)
        if self.reason == CleaningReason.EXACT_DUPLICATE and self.representative_index is None:
            msg = "exact duplicates require representative_index"
            raise ValueError(msg)
        return self


class DatasetCleaningStats(_CleaningModel):
    input_rows: int = Field(ge=0)
    input_text_units: int = Field(ge=0)
    input_comments: int = Field(ge=0)
    input_replies: int = Field(ge=0)
    output_text_units: int = Field(ge=0)
    output_comments: int = Field(ge=0)
    output_replies: int = Field(ge=0)
    removed_by_reason: dict[CleaningReason, int]
    detected_languages: dict[str, int]
    duplicate_groups: int = Field(ge=0)
    largest_duplicate_group: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_counts(self) -> DatasetCleaningStats:
        if self.input_comments + self.input_replies != self.input_text_units:
            msg = "comment and reply inputs must add up to input_text_units"
            raise ValueError(msg)
        if self.output_comments + self.output_replies != self.output_text_units:
            msg = "comment and reply outputs must add up to output_text_units"
            raise ValueError(msg)
        if self.output_text_units + sum(self.removed_by_reason.values()) != self.input_text_units:
            msg = "written and removed counts must add up to input_text_units"
            raise ValueError(msg)
        return self


class DatasetCleaningManifest(_CleaningModel):
    schema_version: int = 1
    source_path: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    split_manifest_path: str
    config: CleaningConfig
    output_path: str
    output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    decisions_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    stats: DatasetCleaningStats
    warnings: list[str]
    created_at: datetime


@dataclass(slots=True)
class _CleaningPassState:
    representatives: dict[str, int]
    duplicate_sizes: Counter[str]
    language_cache: dict[bytes, str]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(source_path: Path, manifest_path: Path) -> DatasetSplitManifest:
    manifest = DatasetSplitManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    expected = manifest.output_sha256.get(SplitName.DEVELOPMENT)
    if _sha256_file(source_path) != expected:
        msg = "dataset checksum does not match development output in split manifest"
        raise ValueError(msg)
    return manifest


def _ensure_unchanged(source_path: Path, expected_sha256: str) -> None:
    if _sha256_file(source_path) != expected_sha256:
        msg = "development dataset changed while cleaning"
        raise ValueError(msg)


def _iter_units(source: BinaryIO) -> Iterator[tuple[int, TextUnit]]:
    unit_index = 0
    for line_number, raw_line in enumerate(source, start=1):
        if not raw_line.strip():
            continue
        try:
            data: Any = json.loads(raw_line.decode("utf-8-sig"))
            comment = ExportedComment.model_validate(data)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            msg = f"invalid development record at line {line_number}: {type(exc).__name__}"
            raise ValueError(msg) from exc
        for unit in exported_comment_to_text_units(comment):
            yield unit_index, unit
            unit_index += 1


def _record_digest(unit: TextUnit) -> str:
    return hashlib.sha256(unit.model_dump_json().encode()).hexdigest()


def _dedup_key(unit: TextUnit, normalized_text_key: str) -> str:
    return f"{unit.text_kind}\x1f{normalized_text_key}"


def _language(text: str, config: CleaningConfig, cache: dict[bytes, str]) -> str:
    if not config.detect_language:
        return "not_detected"
    digest = hashlib.sha256(text.encode()).digest()
    if digest not in cache:
        cache[digest] = detect_language(text)
    return cache[digest]


def _first_pass(
    source_path: Path,
    config: CleaningConfig,
) -> tuple[_CleaningPassState, DatasetCleaningStats]:
    representatives: dict[str, int] = {}
    duplicate_sizes: Counter[str] = Counter()
    language_cache: dict[bytes, str] = {}
    removed: Counter[CleaningReason] = Counter()
    languages: Counter[str] = Counter()
    input_kinds: Counter[TextKind] = Counter()
    output_kinds: Counter[TextKind] = Counter()
    input_rows = 0
    input_units = 0
    with source_path.open("rb") as source:
        for raw_line in source:
            if raw_line.strip():
                input_rows += 1
    with source_path.open("rb") as source:
        for index, unit in _iter_units(source):
            input_units += 1
            input_kinds[unit.text_kind] += 1
            prepared = prepare_text(unit.text, config)
            language = _language(prepared.clean_text, config, language_cache)
            languages[language] += 1
            decision = classify_prepared_text(unit.text, prepared, language, config)
            if not decision.keep:
                assert decision.reason is not None  # noqa: S101 - guaranteed by CleaningDecision
                removed[decision.reason] += 1
                continue
            if config.exact_deduplication:
                dedup_key = _dedup_key(unit, prepared.normalized_text_key)
                if dedup_key in representatives:
                    removed[CleaningReason.EXACT_DUPLICATE] += 1
                    duplicate_sizes[dedup_key] += 1
                    continue
                representatives[dedup_key] = index
                duplicate_sizes[dedup_key] += 1
            output_kinds[unit.text_kind] += 1
    duplicate_groups = [size for size in duplicate_sizes.values() if size > 1]
    stats = DatasetCleaningStats(
        input_rows=input_rows,
        input_text_units=input_units,
        input_comments=input_kinds[TextKind.COMMENT],
        input_replies=input_kinds[TextKind.REPLY],
        output_text_units=sum(output_kinds.values()),
        output_comments=output_kinds[TextKind.COMMENT],
        output_replies=output_kinds[TextKind.REPLY],
        removed_by_reason=dict(removed),
        detected_languages=dict(languages),
        duplicate_groups=len(duplicate_groups),
        largest_duplicate_group=max(duplicate_groups, default=0),
    )
    state = _CleaningPassState(
        representatives=representatives,
        duplicate_sizes=duplicate_sizes,
        language_cache=language_cache,
    )
    return state, stats


def _write_outputs(
    source_path: Path,
    output_path: Path,
    decisions_path: Path,
    *,
    config: CleaningConfig,
    state: _CleaningPassState,
) -> tuple[str, str]:
    output_hash = hashlib.sha256()
    decisions_hash = hashlib.sha256()
    with output_path.open("wb") as output, decisions_path.open("wb") as decisions, source_path.open("rb") as source:
        for index, unit in _iter_units(source):
            prepared = prepare_text(unit.text, config)
            language = _language(prepared.clean_text, config, state.language_cache)
            decision = classify_prepared_text(unit.text, prepared, language, config)
            dedup_key = _dedup_key(unit, prepared.normalized_text_key)
            representative = state.representatives.get(dedup_key)
            reason = decision.reason
            kept = decision.keep
            if kept and config.exact_deduplication and representative != index:
                kept = False
                reason = CleaningReason.EXACT_DUPLICATE
            decision_record = CleaningDecisionRecord(
                record_index=index,
                record_sha256=_record_digest(unit),
                text_kind=unit.text_kind,
                kept=kept,
                reason=reason,
                detected_language=language,
                representative_index=representative if reason == CleaningReason.EXACT_DUPLICATE else None,
            )
            decision_line = f"{decision_record.model_dump_json()}\n".encode()
            decisions.write(decision_line)
            decisions_hash.update(decision_line)
            if not kept:
                continue
            cleaned = CleanedTextUnit(
                **unit.model_dump(),
                clean_text=prepared.clean_text,
                detected_language=language,
                duplicate_count=state.duplicate_sizes.get(dedup_key, 1),
            )
            output_line = f"{cleaned.model_dump_json()}\n".encode()
            output.write(output_line)
            output_hash.update(output_line)
    return output_hash.hexdigest(), decisions_hash.hexdigest()


def clean_development_dataset(
    source_path: Path,
    split_manifest_path: Path,
    output_dir: Path,
    *,
    config: CleaningConfig | None = None,
    force: bool = False,
) -> DatasetCleaningManifest:
    """Clean only the verified development split and persist an auditable result."""
    active_config = config or CleaningConfig()
    split_manifest = _load_manifest(source_path, split_manifest_path)
    expected_rows = split_manifest.stats.written_records.get(SplitName.DEVELOPMENT, 0)
    output_path = output_dir / "development-clean.jsonl"
    decisions_path = output_dir / "cleaning-decisions.jsonl"
    manifest_path = output_dir / "cleaning-manifest.json"
    report_path = output_dir / "cleaning-report.md"
    tables_dir = output_dir / "aggregate-tables"
    removals_path = tables_dir / "removed-by-reason.csv"
    languages_path = tables_dir / "detected-languages.csv"
    artifacts = (output_path, decisions_path, manifest_path, report_path, removals_path, languages_path)
    existing = next((path for path in artifacts if path.exists()), None)
    if existing is not None and not force:
        msg = f"refusing to overwrite existing cleaning artifact: {existing}"
        raise FileExistsError(msg)
    state, stats = _first_pass(source_path, active_config)
    if stats.input_rows != expected_rows:
        msg = f"development row count {stats.input_rows} does not match manifest count {expected_rows}"
        raise ValueError(msg)
    if _sha256_file(source_path) != split_manifest.output_sha256[SplitName.DEVELOPMENT]:
        msg = "development dataset changed while cleaning"
        raise ValueError(msg)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_tmp = output_path.with_suffix(f"{output_path.suffix}.tmp")
    decisions_tmp = decisions_path.with_suffix(f"{decisions_path.suffix}.tmp")
    try:
        output_sha256, decisions_sha256 = _write_outputs(
            source_path,
            output_tmp,
            decisions_tmp,
            config=active_config,
            state=state,
        )
        _ensure_unchanged(source_path, split_manifest.output_sha256[SplitName.DEVELOPMENT])
        output_tmp.replace(output_path)
        decisions_tmp.replace(decisions_path)
    except Exception:
        output_tmp.unlink(missing_ok=True)
        decisions_tmp.unlink(missing_ok=True)
        raise
    retention = stats.output_text_units / stats.input_text_units if stats.input_text_units else 0.0
    warnings = []
    if stats.input_text_units and retention < _LOW_RETENTION_WARNING:
        warnings.append(f"low retained share: {retention:.1%}; review cleaning configuration")
    manifest = DatasetCleaningManifest(
        source_path=str(source_path),
        source_sha256=split_manifest.output_sha256[SplitName.DEVELOPMENT],
        split_manifest_path=str(split_manifest_path),
        config=active_config,
        output_path=str(output_path),
        output_sha256=output_sha256,
        decisions_sha256=decisions_sha256,
        stats=stats,
        warnings=warnings,
        created_at=datetime.now(UTC),
    )
    manifest_path.write_text(f"{manifest.model_dump_json(indent=2)}\n", encoding="utf-8")
    report_path.write_text(_cleaning_markdown(manifest), encoding="utf-8")
    tables_dir.mkdir(parents=True, exist_ok=True)
    _write_aggregate_csv(removals_path, "reason", {str(key): value for key, value in stats.removed_by_reason.items()})
    _write_aggregate_csv(languages_path, "language", stats.detected_languages)
    return manifest


def _write_aggregate_csv(path: Path, label: str, values: dict[str, int]) -> None:
    with path.open("w", encoding="utf-8", newline="") as target:
        csv_writer = writer(target)
        csv_writer.writerow([label, "records"])
        csv_writer.writerows(sorted(values.items()))


def _cleaning_markdown(manifest: DatasetCleaningManifest) -> str:
    stats = manifest.stats
    retention = stats.output_text_units / stats.input_text_units if stats.input_text_units else 0.0
    reasons = "\n".join(f"- `{reason}`: {count}" for reason, count in sorted(stats.removed_by_reason.items()))
    languages = "\n".join(f"- `{language}`: {count}" for language, count in sorted(stats.detected_languages.items()))
    warnings = "\n".join(f"- {warning}" for warning in manifest.warnings) or "- None."
    return f"""# Development cleaning

- Source SHA-256: `{manifest.source_sha256}`
- Output SHA-256: `{manifest.output_sha256}`
- Input rows: {stats.input_rows}
- Input text units: {stats.input_text_units} ({stats.input_comments} comments, {stats.input_replies} replies)
- Retained text units: {stats.output_text_units} ({retention:.1%})
- Retained comments/replies: {stats.output_comments}/{stats.output_replies}
- Exact duplicate groups: {stats.duplicate_groups}
- Largest exact duplicate group: {stats.largest_duplicate_group}

## Removed by reason

{reasons or '- None.'}

## Detected languages

{languages or '- None.'}

## Warnings

{warnings}

Decision records contain only indexes, hashes, roles, language labels, and reasons. This report contains no comment
text, author, URL, query, channel, video ID, or reply content. Semantic near-duplicate removal is not part of this
stage.
"""
