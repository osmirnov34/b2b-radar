import csv
import hashlib
import json
from collections import Counter
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.ml.schemas import ExportedComment

_FORMAT_SAMPLE_BYTES = 128 * 1024
_MIN_TABLE_DIMENSION = 2
_SHORT_TEXT_LENGTH = 5
_DEFAULT_MIN_TEXT_LENGTH = 20
_LONG_TEXT_LENGTH = 1_000
_EXTREME_TEXT_LENGTH = 10_000
_HTML_PREFIXES = ("<!doctype html", "<html", "<head", "<body")
_PROVENANCE_FIELDS = (
    "comment_id",
    "comment_author",
    "comment_published_at",
    "video_id",
    "video_title",
    "video_channel",
    "video_url",
    "search_query",
)


class DatasetFormat(StrEnum):
    JSONL = "jsonl"
    JSON_ARRAY = "json_array"
    JSON_OBJECT = "json_object"
    CSV = "csv"
    TSV = "tsv"
    HTML = "html"
    ZIP = "zip"
    GZIP = "gzip"
    TEXT = "text"
    BINARY = "binary"
    EMPTY = "empty"
    UNKNOWN = "unknown"


class _InspectionModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FormatInspection(_InspectionModel):
    expected: DatasetFormat = DatasetFormat.JSONL
    detected: DatasetFormat
    encoding: str | None = None
    matches: bool
    details: str | None = None


class InspectionError(_InspectionModel):
    line_number: int = Field(ge=1)
    error_type: str
    message: str
    content_hash: str
    fields: list[str] = Field(default_factory=list)
    field_types: dict[str, str] = Field(default_factory=dict)


class TextLengthStats(_InspectionModel):
    minimum: int = Field(ge=0)
    maximum: int = Field(ge=0)
    mean: float = Field(ge=0)
    p50: int = Field(ge=0)
    p75: int = Field(ge=0)
    p90: int = Field(ge=0)
    p95: int = Field(ge=0)
    p99: int = Field(ge=0)
    under_5: int = Field(ge=0)
    under_20: int = Field(ge=0)
    over_1000: int = Field(ge=0)
    over_10000: int = Field(ge=0)


class DatasetInspection(_InspectionModel):
    schema_version: int = 1
    path: str
    size_bytes: int = Field(ge=0)
    sha256: str
    format: FormatInspection
    lines_total: int = Field(ge=0)
    empty_lines: int = Field(ge=0)
    json_valid: int = Field(ge=0)
    json_invalid: int = Field(ge=0)
    non_object_rows: int = Field(ge=0)
    contract_valid: int = Field(ge=0)
    contract_invalid: int = Field(ge=0)
    non_empty_text: int = Field(ge=0)
    missing_fields: dict[str, int]
    unique_authors: int = Field(ge=0)
    unique_videos: int = Field(ge=0)
    unique_channels: int = Field(ge=0)
    unique_queries: int = Field(ge=0)
    unique_comment_ids: int = Field(ge=0)
    duplicate_comment_ids: int = Field(ge=0)
    conflicting_comment_ids: int = Field(ge=0)
    unique_texts: int = Field(ge=0)
    duplicate_texts: int = Field(ge=0)
    duplicate_text_groups: int = Field(ge=0)
    largest_duplicate_text_group: int = Field(ge=0)
    comments_per_video_top: dict[str, int]
    comments_per_channel_top: dict[str, int]
    date_min: datetime | None = None
    date_max: datetime | None = None
    text_lengths: TextLengthStats | None = None
    errors: list[InspectionError]
    error_rate: float = Field(ge=0, le=1)
    expected_records: int | None = Field(default=None, ge=1)
    record_count_matches_expectation: bool | None = None
    critical_errors: list[str]

    @property
    def is_usable(self) -> bool:
        return not self.critical_errors


def _detect_delimited(text: str) -> DatasetFormat | None:
    try:
        dialect = csv.Sniffer().sniff(text[:8192], delimiters=",\t;|")
        rows = list(csv.reader(text[:8192].splitlines(), dialect))
    except csv.Error:
        return None
    if len(rows) < _MIN_TABLE_DIMENSION or len(rows[0]) < _MIN_TABLE_DIMENSION:
        return None
    return DatasetFormat.TSV if dialect.delimiter == "\t" else DatasetFormat.CSV


def detect_dataset_format(path: Path, *, expected: DatasetFormat = DatasetFormat.JSONL) -> FormatInspection:
    with path.open("rb") as source:
        sample = source.read(_FORMAT_SAMPLE_BYTES)
    if not sample:
        detected = DatasetFormat.EMPTY
        return FormatInspection(
            expected=expected,
            detected=detected,
            matches=detected == expected,
            details="empty file",
        )
    if sample.startswith(b"PK\x03\x04"):
        return FormatInspection(expected=expected, detected=DatasetFormat.ZIP, matches=False, details="ZIP signature")
    if sample.startswith(b"\x1f\x8b"):
        return FormatInspection(expected=expected, detected=DatasetFormat.GZIP, matches=False, details="GZIP signature")

    try:
        text = sample.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        return FormatInspection(
            expected=expected,
            detected=DatasetFormat.BINARY,
            matches=False,
            details=f"invalid UTF-8 near byte {exc.start}",
        )

    stripped = text.lstrip()
    lowered = stripped[:256].lower()
    if lowered.startswith(_HTML_PREFIXES):
        detected = DatasetFormat.HTML
    elif stripped.startswith("["):
        detected = DatasetFormat.JSON_ARRAY
    elif stripped.startswith("{"):
        non_empty_lines = [line for line in text.splitlines() if line.strip()]
        line_objects = 0
        for line in non_empty_lines:
            try:
                if isinstance(json.loads(line), dict):
                    line_objects += 1
            except json.JSONDecodeError:
                continue
        looks_like_jsonl = non_empty_lines and line_objects * 2 >= len(non_empty_lines)
        if looks_like_jsonl:
            if len(non_empty_lines) == 1 and path.suffix.lower() == ".json":
                detected = DatasetFormat.JSON_OBJECT
            else:
                detected = DatasetFormat.JSONL
        else:
            detected = DatasetFormat.JSON_OBJECT
    else:
        detected = _detect_delimited(text) or DatasetFormat.TEXT

    details = f"first non-whitespace token: {stripped[:24]!r}" if stripped else "whitespace-only file"
    return FormatInspection(
        expected=expected,
        detected=detected,
        encoding="utf-8-sig" if sample.startswith(b"\xef\xbb\xbf") else "utf-8",
        matches=detected == expected,
        details=details,
    )


def _safe_error(
    line_number: int,
    raw_line: bytes,
    error_type: str,
    message: str,
    data: object | None = None,
) -> InspectionError:
    fields: list[str] = []
    field_types: dict[str, str] = {}
    if isinstance(data, dict):
        fields = sorted(str(key) for key in data)
        field_types = {str(key): type(value).__name__ for key, value in data.items()}
    return InspectionError(
        line_number=line_number,
        error_type=error_type,
        message=message,
        content_hash=hashlib.sha256(raw_line).hexdigest()[:16],
        fields=fields,
        field_types=field_types,
    )


def _validation_message(error: ValidationError) -> str:
    """Render validation failures without Pydantic's raw `input`, which may contain private text."""
    parts: list[str] = []
    for detail in error.errors(include_url=False, include_context=False, include_input=False):
        location = ".".join(str(item) for item in detail["loc"]) or "record"
        parts.append(f"{location}: {detail['msg']} ({detail['type']})")
    return "; ".join(parts)


def _length_stats(lengths: list[int]) -> TextLengthStats | None:
    if not lengths:
        return None
    ordered = sorted(lengths)

    def percentile(fraction: float) -> int:
        return ordered[round((len(ordered) - 1) * fraction)]

    return TextLengthStats(
        minimum=ordered[0],
        maximum=ordered[-1],
        mean=sum(ordered) / len(ordered),
        p50=percentile(0.50),
        p75=percentile(0.75),
        p90=percentile(0.90),
        p95=percentile(0.95),
        p99=percentile(0.99),
        under_5=sum(length < _SHORT_TEXT_LENGTH for length in ordered),
        under_20=sum(length < _DEFAULT_MIN_TEXT_LENGTH for length in ordered),
        over_1000=sum(length > _LONG_TEXT_LENGTH for length in ordered),
        over_10000=sum(length > _EXTREME_TEXT_LENGTH for length in ordered),
    )


def _top_counts(counter: Counter[str], limit: int = 20) -> dict[str, int]:
    return dict(counter.most_common(limit))


def inspect_comments_jsonl(
    path: Path,
    *,
    error_sample_limit: int = 100,
    max_error_rate: float = 0.01,
    expected_records: int | None = None,
    expected_records_tolerance: float = 0.10,
) -> DatasetInspection:
    if not 0 <= max_error_rate <= 1:
        msg = "max_error_rate must be between 0 and 1"
        raise ValueError(msg)
    if not 0 <= expected_records_tolerance <= 1:
        msg = "expected_records_tolerance must be between 0 and 1"
        raise ValueError(msg)

    format_result = detect_dataset_format(path)
    file_hash = hashlib.sha256()
    errors: list[InspectionError] = []
    missing = Counter[str]()
    authors: set[str] = set()
    videos: set[str] = set()
    channels: set[str] = set()
    queries: set[str] = set()
    id_text_hashes: dict[str, str] = {}
    duplicate_ids = 0
    conflicting_ids = 0
    text_counts: Counter[str] = Counter()
    video_counts: Counter[str] = Counter()
    channel_counts: Counter[str] = Counter()
    lengths: list[int] = []
    date_min: datetime | None = None
    date_max: datetime | None = None
    lines_total = empty_lines = json_valid = json_invalid = non_objects = 0
    contract_valid = contract_invalid = non_empty_text = 0

    with path.open("rb") as source:
        for line_number, raw_line in enumerate(source, start=1):
            file_hash.update(raw_line)
            lines_total += 1
            if not raw_line.strip():
                empty_lines += 1
                continue
            try:
                decoded = raw_line.decode("utf-8-sig")
            except UnicodeDecodeError as exc:
                json_invalid += 1
                if len(errors) < error_sample_limit:
                    errors.append(
                        _safe_error(
                            line_number,
                            raw_line,
                            "invalid_utf8",
                            f"invalid UTF-8 near byte {exc.start}",
                        ),
                    )
                continue
            try:
                data: Any = json.loads(decoded)
                json_valid += 1
            except json.JSONDecodeError as exc:
                json_invalid += 1
                if len(errors) < error_sample_limit:
                    errors.append(_safe_error(line_number, raw_line, "json_decode", exc.msg))
                continue
            if not isinstance(data, dict):
                non_objects += 1
                if len(errors) < error_sample_limit:
                    errors.append(_safe_error(line_number, raw_line, "non_object", type(data).__name__))
                continue
            try:
                comment = ExportedComment.model_validate(data)
                contract_valid += 1
            except ValidationError as exc:
                contract_invalid += 1
                if len(errors) < error_sample_limit:
                    errors.append(_safe_error(line_number, raw_line, "schema", _validation_message(exc), data))
                continue

            for field in _PROVENANCE_FIELDS:
                if not getattr(comment, field):
                    missing[field] += 1
            text = comment.comment_text.strip()
            if not text:
                continue
            non_empty_text += 1
            lengths.append(len(text))
            normalized_hash = hashlib.sha256(" ".join(text.casefold().split()).encode()).hexdigest()
            text_counts[normalized_hash] += 1
            if comment.comment_id:
                previous_hash = id_text_hashes.get(comment.comment_id)
                if previous_hash is not None:
                    duplicate_ids += 1
                    if previous_hash != normalized_hash:
                        conflicting_ids += 1
                else:
                    id_text_hashes[comment.comment_id] = normalized_hash
            if comment.comment_author:
                authors.add(comment.comment_author)
            if comment.video_id:
                videos.add(comment.video_id)
                video_counts[comment.video_id] += 1
            if comment.video_channel:
                channels.add(comment.video_channel)
                channel_counts[comment.video_channel] += 1
            if comment.search_query:
                queries.add(comment.search_query)
            if comment.comment_published_at is not None:
                date_min = (
                    comment.comment_published_at
                    if date_min is None
                    else min(date_min, comment.comment_published_at)
                )
                date_max = (
                    comment.comment_published_at
                    if date_max is None
                    else max(date_max, comment.comment_published_at)
                )

    checked_rows = json_invalid + non_objects + contract_valid + contract_invalid
    invalid_rows = json_invalid + non_objects + contract_invalid
    error_rate = invalid_rows / checked_rows if checked_rows else 0.0
    duplicate_groups = [count for count in text_counts.values() if count > 1]
    record_count_matches: bool | None = None
    if expected_records is not None:
        difference = abs(contract_valid - expected_records) / expected_records
        record_count_matches = difference <= expected_records_tolerance

    critical: list[str] = []
    if not format_result.matches:
        critical.append(f"expected {format_result.expected}, detected {format_result.detected}")
    if contract_valid == 0:
        critical.append("no records match the ExportedComment contract")
    if error_rate > max_error_rate:
        critical.append(f"row error rate {error_rate:.2%} exceeds allowed {max_error_rate:.2%}")
    if record_count_matches is False:
        critical.append(f"record count {contract_valid} differs from expected {expected_records}")

    return DatasetInspection(
        path=str(path),
        size_bytes=path.stat().st_size,
        sha256=file_hash.hexdigest(),
        format=format_result,
        lines_total=lines_total,
        empty_lines=empty_lines,
        json_valid=json_valid,
        json_invalid=json_invalid,
        non_object_rows=non_objects,
        contract_valid=contract_valid,
        contract_invalid=contract_invalid,
        non_empty_text=non_empty_text,
        missing_fields={field: missing[field] for field in _PROVENANCE_FIELDS},
        unique_authors=len(authors),
        unique_videos=len(videos),
        unique_channels=len(channels),
        unique_queries=len(queries),
        unique_comment_ids=len(id_text_hashes),
        duplicate_comment_ids=duplicate_ids,
        conflicting_comment_ids=conflicting_ids,
        unique_texts=len(text_counts),
        duplicate_texts=sum(count - 1 for count in duplicate_groups),
        duplicate_text_groups=len(duplicate_groups),
        largest_duplicate_text_group=max(duplicate_groups, default=0),
        comments_per_video_top=_top_counts(video_counts),
        comments_per_channel_top=_top_counts(channel_counts),
        date_min=date_min,
        date_max=date_max,
        text_lengths=_length_stats(lengths),
        errors=errors,
        error_rate=error_rate,
        expected_records=expected_records,
        record_count_matches_expectation=record_count_matches,
        critical_errors=critical,
    )
