"""Privacy-safe readers for canonical analytical artifacts."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from src.analytics.contracts import FeedbackRecord
from src.data_formats import DatasetFormat, detect_dataset_format

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


class FeedbackArtifactError(ValueError):
    """Canonical artifact failure that never includes feedback text."""

    def __init__(self, line_number: int, error_type: str, message: str, raw_line: bytes) -> None:
        self.line_number = line_number
        self.error_type = error_type
        self.content_hash = hashlib.sha256(raw_line).hexdigest()[:16]
        super().__init__(f"line {line_number}: {error_type}: {message}; content_hash={self.content_hash}")


def _validation_message(error: ValidationError) -> str:
    parts: list[str] = []
    for detail in error.errors(include_url=False, include_context=False, include_input=False):
        location = ".".join(str(item) for item in detail["loc"]) or "record"
        parts.append(f"{location}: {detail['msg']} ({detail['type']})")
    return "; ".join(parts)


def parse_feedback_line(raw_line: bytes, *, line_number: int) -> FeedbackRecord:
    """Parse one canonical JSON object using sanitized diagnostics."""
    try:
        decoded = raw_line.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise FeedbackArtifactError(
            line_number,
            "invalid_utf8",
            f"invalid UTF-8 at byte {exc.start}",
            raw_line,
        ) from exc
    try:
        data: Any = json.loads(decoded)
    except json.JSONDecodeError as exc:
        raise FeedbackArtifactError(line_number, "json_decode", exc.msg, raw_line) from exc
    if not isinstance(data, dict):
        raise FeedbackArtifactError(line_number, "non_object", type(data).__name__, raw_line)
    try:
        return FeedbackRecord.model_validate(data)
    except ValidationError as exc:
        raise FeedbackArtifactError(line_number, "schema", _validation_message(exc), raw_line) from exc


def iter_feedback_records(path: Path) -> Iterator[FeedbackRecord]:
    """Stream a canonical UTF-8 JSONL artifact without retaining private text."""
    inspection = detect_dataset_format(path, expected=DatasetFormat.JSONL)
    if not inspection.matches or inspection.encoding not in {"utf-8", "utf-8-sig"}:
        msg = f"expected UTF-8 JSONL, detected {inspection.detected.value}"
        raise ValueError(msg)
    with path.open("rb") as source:
        for line_number, raw_line in enumerate(source, start=1):
            if raw_line.strip():
                yield parse_feedback_line(raw_line, line_number=line_number)
