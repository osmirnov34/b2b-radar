"""Strict source contract and privacy-safe reader for Yandex Maps reviews."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date, datetime
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from src.data_formats import DatasetFormat, FormatInspection, detect_dataset_format

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

_REVIEW_ID_PATTERN = re.compile(r"^[0-9a-f]{16}$")
_ORGANIZATION_ID_PATTERN = re.compile(r"^[0-9]+$")
_ISO_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_YANDEX_HOSTS = {"yandex.ru", "www.yandex.ru"}


class YandexReview(BaseModel):
    """One unmodified Yandex Maps review with explicit organization provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    review_id: str = Field(pattern=_REVIEW_ID_PATTERN.pattern)
    review_text: str
    rating: int = Field(ge=1, le=5, strict=True)
    author: str = ""
    published_at: date
    org_id: str = Field(pattern=_ORGANIZATION_ID_PATTERN.pattern)
    org_name: str = Field(min_length=1)
    org_url: str = Field(min_length=1)
    search_query: str = ""

    @field_validator("review_id", "org_id", "org_name", "org_url", mode="before")
    @classmethod
    def strip_identity_fields(cls, value: object) -> object:
        """Normalize harmless surrounding whitespace while keeping identities strict."""
        return value.strip() if isinstance(value, str) else value

    @field_validator("author", "search_query", mode="before")
    @classmethod
    def normalize_nullable_optional_strings(cls, value: object) -> object:
        """Normalize source nulls without inventing identity or text values."""
        return "" if value is None else value

    @field_validator("published_at", mode="before")
    @classmethod
    def require_iso_calendar_date(cls, value: object) -> object:
        """Accept the source's YYYY-MM-DD strings and already parsed dates only."""
        if isinstance(value, datetime):
            msg = "published_at must be a calendar date, not a datetime"
            raise ValueError(msg)  # noqa: TRY004 -- Pydantic validators require ValueError for value diagnostics.
        if isinstance(value, date):
            return value
        if isinstance(value, str) and _ISO_DATE_PATTERN.fullmatch(value.strip()):
            return value.strip()
        msg = "published_at must use YYYY-MM-DD"
        raise ValueError(msg)

    @field_validator("org_url")
    @classmethod
    def require_yandex_https_url(cls, value: str) -> str:
        """Reject unrelated or insecure organization URLs."""
        parsed = urlparse(value)
        if parsed.scheme != "https" or parsed.hostname not in _YANDEX_HOSTS:
            msg = "org_url must be an HTTPS yandex.ru URL"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def require_matching_organization_url(self) -> YandexReview:
        """Bind the organization URL to the explicit Yandex organization ID."""
        if f"/org/{self.org_id}/" not in urlparse(self.org_url).path:
            msg = "org_url does not contain org_id"
            raise ValueError(msg)
        return self


class YandexReviewFormatError(ValueError):
    """Raised before reading records when the source is not UTF-8 JSONL."""

    def __init__(self, inspection: FormatInspection) -> None:
        self.inspection = inspection
        detail = f"; {inspection.details}" if inspection.details else ""
        super().__init__(f"expected UTF-8 JSONL, detected {inspection.detected.value}{detail}")


class YandexReviewLineError(ValueError):
    """A sanitized record failure that never includes source values."""

    def __init__(
        self,
        *,
        line_number: int,
        error_type: str,
        message: str,
        raw_line: bytes,
        data: object | None = None,
    ) -> None:
        self.line_number = line_number
        self.error_type = error_type
        self.content_hash = hashlib.sha256(raw_line).hexdigest()[:16]
        self.fields: tuple[str, ...] = ()
        self.field_types: dict[str, str] = {}
        if isinstance(data, dict):
            self.fields = tuple(sorted(str(key) for key in data))
            self.field_types = {str(key): type(value).__name__ for key, value in data.items()}
        super().__init__(f"line {line_number}: {error_type}: {message}; content_hash={self.content_hash}")


def sanitized_validation_message(error: ValidationError) -> str:
    """Render Pydantic errors without values from private source fields."""
    parts: list[str] = []
    for detail in error.errors(include_url=False, include_context=False, include_input=False):
        location = ".".join(str(item) for item in detail["loc"]) or "record"
        parts.append(f"{location}: {detail['msg']} ({detail['type']})")
    return "; ".join(parts)


def parse_yandex_review_line(raw_line: bytes, *, line_number: int) -> YandexReview:
    """Parse one line and raise only privacy-safe diagnostics."""
    try:
        decoded = raw_line.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise YandexReviewLineError(
            line_number=line_number,
            error_type="invalid_utf8",
            message=f"invalid UTF-8 at byte {exc.start}",
            raw_line=raw_line,
        ) from exc
    try:
        data: Any = json.loads(decoded)
    except json.JSONDecodeError as exc:
        raise YandexReviewLineError(
            line_number=line_number,
            error_type="json_decode",
            message=exc.msg,
            raw_line=raw_line,
        ) from exc
    if not isinstance(data, dict):
        raise YandexReviewLineError(
            line_number=line_number,
            error_type="non_object",
            message=type(data).__name__,
            raw_line=raw_line,
        )
    try:
        return YandexReview.model_validate(data)
    except ValidationError as exc:
        raise YandexReviewLineError(
            line_number=line_number,
            error_type="schema",
            message=sanitized_validation_message(exc),
            raw_line=raw_line,
            data=data,
        ) from exc


def iter_yandex_reviews(path: Path) -> Iterator[YandexReview]:
    """Stream a verified source without loading private records into one collection."""
    inspection = detect_dataset_format(path, expected=DatasetFormat.JSONL)
    if not inspection.matches or inspection.encoding not in {"utf-8", "utf-8-sig"}:
        raise YandexReviewFormatError(inspection)
    with path.open("rb") as source:
        for line_number, raw_line in enumerate(source, start=1):
            if raw_line.strip():
                yield parse_yandex_review_line(raw_line, line_number=line_number)
