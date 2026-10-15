"""Source-neutral dataset format detection."""

from __future__ import annotations

import codecs
import csv
import json
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from pathlib import Path

_FORMAT_SAMPLE_BYTES = 128 * 1024
_MIN_TABLE_DIMENSION = 2
_HTML_PREFIXES = ("<!doctype html", "<html", "<head", "<body")


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


class FormatInspection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    expected: DatasetFormat = DatasetFormat.JSONL
    detected: DatasetFormat
    encoding: str | None = None
    matches: bool
    details: str | None = None


def _detect_delimited(text: str) -> DatasetFormat | None:
    try:
        dialect = csv.Sniffer().sniff(text[:8192], delimiters=",\t;|")
        rows = list(csv.reader(text[:8192].splitlines(), dialect))
    except csv.Error:
        return None
    if len(rows) < _MIN_TABLE_DIMENSION or len(rows[0]) < _MIN_TABLE_DIMENSION:
        return None
    return DatasetFormat.TSV if dialect.delimiter == "\t" else DatasetFormat.CSV


def _detect_text_format(path: Path, text: str) -> DatasetFormat:
    stripped = text.lstrip()
    lowered = stripped[:256].lower()
    if lowered.startswith(_HTML_PREFIXES):
        return DatasetFormat.HTML
    if stripped.startswith("["):
        return DatasetFormat.JSON_ARRAY
    if not stripped.startswith("{"):
        return _detect_delimited(text) or DatasetFormat.TEXT
    non_empty_lines = [line for line in text.splitlines() if line.strip()]
    line_objects = 0
    for line in non_empty_lines:
        try:
            if isinstance(json.loads(line), dict):
                line_objects += 1
        except json.JSONDecodeError:
            continue
    looks_like_jsonl = non_empty_lines and line_objects * 2 >= len(non_empty_lines)
    if not looks_like_jsonl:
        return DatasetFormat.JSON_OBJECT
    single_json_file = len(non_empty_lines) == 1 and path.suffix.lower() == ".json"
    return DatasetFormat.JSON_OBJECT if single_json_file else DatasetFormat.JSONL


def detect_dataset_format(path: Path, *, expected: DatasetFormat = DatasetFormat.JSONL) -> FormatInspection:
    """Detect a dataset container and text encoding from a bounded prefix."""
    with path.open("rb") as source:
        sample = source.read(_FORMAT_SAMPLE_BYTES)
    signature_formats = ((b"PK\x03\x04", DatasetFormat.ZIP), (b"\x1f\x8b", DatasetFormat.GZIP))
    if not sample:
        return FormatInspection(expected=expected, detected=DatasetFormat.EMPTY, matches=False, details="empty file")
    for signature, detected in signature_formats:
        if sample.startswith(signature):
            return FormatInspection(
                expected=expected,
                detected=detected,
                matches=detected == expected,
                details=f"{detected.value.upper()} signature",
            )
    try:
        decoder = codecs.getincrementaldecoder("utf-8-sig")()
        text = decoder.decode(sample, final=False)
    except UnicodeDecodeError as exc:
        return FormatInspection(
            expected=expected,
            detected=DatasetFormat.BINARY,
            matches=False,
            details=f"invalid UTF-8 near byte {exc.start}",
        )

    detected = _detect_text_format(path, text)
    stripped = text.lstrip()
    details = f"first non-whitespace token: {stripped[:24]!r}" if stripped else "whitespace-only file"
    return FormatInspection(
        expected=expected,
        detected=detected,
        encoding="utf-8-sig" if sample.startswith(b"\xef\xbb\xbf") else "utf-8",
        matches=detected == expected,
        details=details,
    )
