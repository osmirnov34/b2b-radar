"""Privacy-safe, streaming structural audit for Yandex Maps review JSONL."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.data_formats import DatasetFormat, FormatInspection, detect_dataset_format
from src.sources.yandex_reviews import YandexReview, sanitized_validation_message

if TYPE_CHECKING:
    from pathlib import Path

_EXPECTED_FIELDS = (
    "review_id", "review_text", "rating", "author", "published_at",
    "org_id", "org_name", "org_url", "search_query",
)
_REQUIRED_FIELDS = ("review_id", "review_text", "rating", "published_at", "org_id", "org_name", "org_url")
_MINIMUM_RATING = 1
_MAXIMUM_RATING = 5
_SHORT_TEXT_LENGTH = 20
_LONG_TEXT_LENGTH = 1_000


class _AuditModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class YandexReviewAuditConfig(_AuditModel):
    """Reproducible thresholds for one source audit."""

    reference_date: date | None = None
    minimum_reviews_for_comparison: int = Field(default=30, ge=1)
    max_error_rate: float = Field(default=0.01, ge=0, le=1)
    error_sample_limit: int = Field(default=100, ge=0)
    top_organization_limit: int = Field(default=20, ge=0)


class YandexReviewAuditError(_AuditModel):
    """One sanitized input error without source values."""

    line_number: int = Field(ge=1)
    error_type: str
    message: str
    content_hash: str = Field(pattern=r"^[0-9a-f]{16}$")
    fields: tuple[str, ...] = ()
    field_types: dict[str, str] = Field(default_factory=dict)


class NumericSummary(_AuditModel):
    minimum: int = Field(ge=0)
    p50: int = Field(ge=0)
    p95: int = Field(ge=0)
    maximum: int = Field(ge=0)
    mean: float = Field(ge=0)


class OrganizationReviewCount(_AuditModel):
    organization_id: str
    organization_name: str
    reviews: int = Field(ge=1)


class YandexReviewAudit(_AuditModel):
    """Aggregate structural and quality report for one immutable source file."""

    schema_version: int = 1
    source_path: str
    source_size_bytes: int = Field(ge=0)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    audited_at: datetime
    reference_date: date
    format: FormatInspection
    lines_total: int = Field(ge=0)
    empty_lines: int = Field(ge=0)
    json_valid: int = Field(ge=0)
    json_invalid: int = Field(ge=0)
    non_object_rows: int = Field(ge=0)
    contract_valid: int = Field(ge=0)
    contract_invalid: int = Field(ge=0)
    error_rate: float = Field(ge=0, le=1)
    field_presence: dict[str, int]
    field_types: dict[str, dict[str, int]]
    unknown_fields: dict[str, int]
    missing_required_fields: dict[str, int]
    empty_review_texts: int = Field(ge=0)
    text_lengths: NumericSummary | None = None
    texts_under_20: int = Field(ge=0)
    texts_over_1000: int = Field(ge=0)
    unique_review_ids: int = Field(ge=0)
    duplicate_review_ids: int = Field(ge=0)
    conflicting_review_ids: int = Field(ge=0)
    unique_texts: int = Field(ge=0)
    duplicate_texts: int = Field(ge=0)
    duplicate_text_groups: int = Field(ge=0)
    largest_duplicate_text_group: int = Field(ge=0)
    rating_distribution: dict[int, int]
    missing_ratings: int = Field(ge=0)
    invalid_ratings: int = Field(ge=0)
    published_at_min: date | None = None
    published_at_max: date | None = None
    missing_dates: int = Field(ge=0)
    invalid_dates: int = Field(ge=0)
    future_dates: int = Field(ge=0)
    unique_organizations: int = Field(ge=0)
    unique_organization_names: int = Field(ge=0)
    unique_organization_urls: int = Field(ge=0)
    organizations_with_multiple_names: int = Field(ge=0)
    organizations_with_multiple_urls: int = Field(ge=0)
    organization_names_with_multiple_ids: int = Field(ge=0)
    invalid_organization_urls: int = Field(ge=0)
    organization_id_url_mismatches: int = Field(ge=0)
    organization_review_counts: NumericSummary | None = None
    organizations_at_or_above_minimum_reviews: int = Field(ge=0)
    minimum_reviews_for_comparison: int = Field(ge=1)
    top_organizations: tuple[OrganizationReviewCount, ...]
    unique_search_queries: int = Field(ge=0)
    missing_search_queries: int = Field(ge=0)
    errors: tuple[YandexReviewAuditError, ...]
    warnings: tuple[str, ...]
    critical_errors: tuple[str, ...]

    @property
    def is_usable(self) -> bool:
        return not self.critical_errors


def _field_type(value: object) -> str:
    checks = (
        (value is None, "null"),
        (isinstance(value, bool), "boolean"),
        (isinstance(value, int), "integer"),
        (isinstance(value, float), "number"),
        (isinstance(value, str), "string"),
        (isinstance(value, list), "array"),
        (isinstance(value, dict), "object"),
    )
    return next((label for matches, label in checks if matches), type(value).__name__)


def _safe_error(
    line_number: int,
    raw_line: bytes,
    error_type: str,
    message: str,
    data: object | None = None,
) -> YandexReviewAuditError:
    fields: tuple[str, ...] = ()
    field_types: dict[str, str] = {}
    if isinstance(data, dict):
        fields = tuple(sorted(str(key) for key in data))
        field_types = {str(key): _field_type(value) for key, value in data.items()}
    return YandexReviewAuditError(
        line_number=line_number,
        error_type=error_type,
        message=message,
        content_hash=hashlib.sha256(raw_line).hexdigest()[:16],
        fields=fields,
        field_types=field_types,
    )


def _numeric_summary(values: list[int]) -> NumericSummary | None:
    if not values:
        return None
    ordered = sorted(values)

    def percentile(fraction: float) -> int:
        return ordered[round((len(ordered) - 1) * fraction)]

    return NumericSummary(
        minimum=ordered[0], p50=percentile(0.50), p95=percentile(0.95),
        maximum=ordered[-1], mean=sum(ordered) / len(ordered),
    )


def _text_hash(text: str) -> str:
    return hashlib.sha256(" ".join(text.casefold().split()).encode()).hexdigest()


def _non_empty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _parse_date(value: object) -> date | None:
    if not _non_empty_string(value):
        return None
    try:
        return date.fromisoformat(str(value).strip())
    except ValueError:
        return None


def _valid_org_url(value: object) -> bool:
    if not _non_empty_string(value):
        return False
    parsed = urlparse(str(value).strip())
    return parsed.scheme == "https" and parsed.hostname in {"yandex.ru", "www.yandex.ru"}


@dataclass
class _AuditAccumulator:
    config: YandexReviewAuditConfig
    reference_date: date
    source_hash: Any = field(default_factory=hashlib.sha256)
    field_presence: Counter[str] = field(default_factory=Counter)
    field_types: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    unknown_fields: Counter[str] = field(default_factory=Counter)
    missing_required: Counter[str] = field(default_factory=Counter)
    errors: list[YandexReviewAuditError] = field(default_factory=list)
    review_ids: dict[str, tuple[str, str]] = field(default_factory=dict)
    texts: Counter[str] = field(default_factory=Counter)
    org_names: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    org_urls: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    name_org_ids: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    org_counts: Counter[str] = field(default_factory=Counter)
    org_display_names: dict[str, str] = field(default_factory=dict)
    ratings: Counter[int] = field(default_factory=Counter)
    queries: set[str] = field(default_factory=set)
    lengths: list[int] = field(default_factory=list)
    dates: list[date] = field(default_factory=list)
    lines_total: int = 0
    empty_lines: int = 0
    json_valid: int = 0
    json_invalid: int = 0
    non_objects: int = 0
    contract_valid: int = 0
    contract_invalid: int = 0
    empty_texts: int = 0
    duplicate_ids: int = 0
    conflicting_ids: int = 0
    missing_ratings: int = 0
    invalid_ratings: int = 0
    missing_dates: int = 0
    invalid_dates: int = 0
    future_dates: int = 0
    invalid_urls: int = 0
    url_id_mismatches: int = 0
    missing_queries: int = 0

    def observe(self, line_number: int, raw_line: bytes) -> None:
        self.source_hash.update(raw_line)
        self.lines_total += 1
        if not raw_line.strip():
            self.empty_lines += 1
            return
        try:
            data: Any = json.loads(raw_line.decode("utf-8-sig"))
        except UnicodeDecodeError as exc:
            self.json_invalid += 1
            self._add_error(line_number, raw_line, "invalid_utf8", f"invalid UTF-8 at byte {exc.start}")
            return
        except json.JSONDecodeError as exc:
            self.json_invalid += 1
            self._add_error(line_number, raw_line, "json_decode", exc.msg)
            return
        self.json_valid += 1
        if not isinstance(data, dict):
            self.non_objects += 1
            self._add_error(line_number, raw_line, "non_object", type(data).__name__)
            return
        self._observe_mapping(line_number, raw_line, data)

    def _add_error(self, line: int, raw: bytes, kind: str, message: str, data: object | None = None) -> None:
        if len(self.errors) < self.config.error_sample_limit:
            self.errors.append(_safe_error(line, raw, kind, message, data))

    def _observe_mapping(self, line: int, raw: bytes, data: dict[str, Any]) -> None:
        self._observe_fields(data)
        self._observe_rating(data.get("rating"))
        self._observe_date(data.get("published_at"))
        self._observe_raw_url(data)
        try:
            review = YandexReview.model_validate(data)
        except ValidationError as exc:
            self.contract_invalid += 1
            self._add_error(line, raw, "schema", sanitized_validation_message(exc), data)
            return
        self.contract_valid += 1
        self._observe_valid_record(review)

    def _observe_fields(self, data: dict[str, Any]) -> None:
        for key, value in data.items():
            name = str(key)
            self.field_presence[name] += 1
            self.field_types[name][_field_type(value)] += 1
            if name not in _EXPECTED_FIELDS:
                self.unknown_fields[name] += 1
        self.missing_required.update(name for name in _REQUIRED_FIELDS if name not in data)

    def _observe_rating(self, value: object) -> None:
        if value is None:
            self.missing_ratings += 1
        elif type(value) is int and _MINIMUM_RATING <= value <= _MAXIMUM_RATING:
            self.ratings[value] += 1
        else:
            self.invalid_ratings += 1

    def _observe_date(self, value: object) -> None:
        if value in (None, ""):
            self.missing_dates += 1
            return
        parsed = _parse_date(value)
        if parsed is None:
            self.invalid_dates += 1
            return
        self.dates.append(parsed)
        self.future_dates += parsed > self.reference_date

    def _observe_raw_url(self, data: dict[str, Any]) -> None:
        if "org_url" not in data:
            return
        org_id = data.get("org_id")
        org_url = data["org_url"]
        if not _valid_org_url(org_url):
            self.invalid_urls += 1
            return
        if _non_empty_string(org_id) and f"/org/{str(org_id).strip()}/" not in urlparse(str(org_url)).path:
            self.url_id_mismatches += 1

    def _observe_valid_record(self, review: YandexReview) -> None:
        review_id = review.review_id
        text = review.review_text.strip()
        org_id = review.org_id
        org_name = review.org_name
        org_url = review.org_url
        fingerprint = (_text_hash(text), org_id)
        previous = self.review_ids.get(review_id)
        self.duplicate_ids += previous is not None
        self.conflicting_ids += previous is not None and previous != fingerprint
        self.review_ids.setdefault(review_id, fingerprint)
        self.empty_texts += not text
        self.lengths.append(len(text))
        self.texts[fingerprint[0]] += 1
        self.org_names[org_id].add(org_name)
        self.org_urls[org_id].add(org_url)
        self.name_org_ids[org_name.casefold()].add(org_id)
        self.org_counts[org_id] += 1
        self.org_display_names.setdefault(org_id, org_name)
        query = review.search_query
        if _non_empty_string(query):
            self.queries.add(str(query).strip())
        else:
            self.missing_queries += 1

    def build(self, path: Path, format_result: FormatInspection) -> YandexReviewAudit:
        checked = self.json_invalid + self.non_objects + self.contract_valid + self.contract_invalid
        invalid = self.json_invalid + self.non_objects + self.contract_invalid
        error_rate = invalid / checked if checked else 0.0
        duplicate_groups = [count for count in self.texts.values() if count > 1]
        org_sizes = list(self.org_counts.values())
        multiple_names = sum(len(values) > 1 for values in self.org_names.values())
        multiple_urls = sum(len(values) > 1 for values in self.org_urls.values())
        shared_names = sum(len(values) > 1 for values in self.name_org_ids.values())
        return YandexReviewAudit(
            source_path=str(path), source_size_bytes=path.stat().st_size,
            source_sha256=self.source_hash.hexdigest(), audited_at=datetime.now(UTC),
            reference_date=self.reference_date, format=format_result, lines_total=self.lines_total,
            empty_lines=self.empty_lines, json_valid=self.json_valid, json_invalid=self.json_invalid,
            non_object_rows=self.non_objects, contract_valid=self.contract_valid,
            contract_invalid=self.contract_invalid, error_rate=error_rate,
            field_presence=dict(sorted(self.field_presence.items())),
            field_types={name: dict(sorted(counts.items())) for name, counts in sorted(self.field_types.items())},
            unknown_fields=dict(sorted(self.unknown_fields.items())),
            missing_required_fields={name: self.missing_required[name] for name in _REQUIRED_FIELDS},
            empty_review_texts=self.empty_texts, text_lengths=_numeric_summary(self.lengths),
            texts_under_20=sum(length < _SHORT_TEXT_LENGTH for length in self.lengths),
            texts_over_1000=sum(length > _LONG_TEXT_LENGTH for length in self.lengths),
            unique_review_ids=len(self.review_ids), duplicate_review_ids=self.duplicate_ids,
            conflicting_review_ids=self.conflicting_ids, unique_texts=len(self.texts),
            duplicate_texts=sum(count - 1 for count in duplicate_groups),
            duplicate_text_groups=len(duplicate_groups), largest_duplicate_text_group=max(duplicate_groups, default=0),
            rating_distribution=dict(sorted(self.ratings.items())), missing_ratings=self.missing_ratings,
            invalid_ratings=self.invalid_ratings, published_at_min=min(self.dates, default=None),
            published_at_max=max(self.dates, default=None), missing_dates=self.missing_dates,
            invalid_dates=self.invalid_dates, future_dates=self.future_dates,
            unique_organizations=len(self.org_names), unique_organization_names=len(self.name_org_ids),
            unique_organization_urls=len({url for values in self.org_urls.values() for url in values}),
            organizations_with_multiple_names=multiple_names, organizations_with_multiple_urls=multiple_urls,
            organization_names_with_multiple_ids=shared_names, invalid_organization_urls=self.invalid_urls,
            organization_id_url_mismatches=self.url_id_mismatches,
            organization_review_counts=_numeric_summary(org_sizes),
            organizations_at_or_above_minimum_reviews=sum(
                count >= self.config.minimum_reviews_for_comparison for count in org_sizes
            ),
            minimum_reviews_for_comparison=self.config.minimum_reviews_for_comparison,
            top_organizations=self._top_organizations(), unique_search_queries=len(self.queries),
            missing_search_queries=self.missing_queries, errors=tuple(self.errors),
            warnings=tuple(self._warnings(multiple_names, multiple_urls, shared_names)),
            critical_errors=tuple(self._critical(format_result, error_rate)),
        )

    def _top_organizations(self) -> tuple[OrganizationReviewCount, ...]:
        return tuple(
            OrganizationReviewCount(
                organization_id=org_id, organization_name=self.org_display_names[org_id], reviews=count,
            )
            for org_id, count in self.org_counts.most_common(self.config.top_organization_limit)
        )

    def _warnings(self, multiple_names: int, multiple_urls: int, shared_names: int) -> list[str]:
        warnings: list[str] = []
        missing_rating_values = sorted(set(range(_MINIMUM_RATING, _MAXIMUM_RATING + 1)) - self.ratings.keys())
        entries = (
            (self.duplicate_ids > 0, f"duplicate review IDs observed: {self.duplicate_ids}"),
            (bool(self.unknown_fields), f"unknown fields observed: {', '.join(sorted(self.unknown_fields))}"),
            (bool(missing_rating_values), f"rating values absent from dataset: {missing_rating_values}"),
            (self.future_dates > 0, f"reviews dated after {self.reference_date}: {self.future_dates}"),
            (multiple_names > 0, f"organization IDs with multiple names: {multiple_names}"),
            (multiple_urls > 0, f"organization IDs with multiple URLs: {multiple_urls}"),
            (shared_names > 0, f"organization names shared by multiple IDs: {shared_names}"),
            (
                self.invalid_urls + self.url_id_mismatches > 0,
                f"organization URL issues: invalid={self.invalid_urls}, id_mismatch={self.url_id_mismatches}",
            ),
        )
        warnings.extend(message for include, message in entries if include)
        return warnings

    def _critical(self, format_result: FormatInspection, error_rate: float) -> list[str]:
        critical: list[str] = []
        entries = (
            (not format_result.matches, f"expected JSONL, detected {format_result.detected.value}"),
            (self.contract_valid == 0, "no records match the expected Yandex review structure"),
            (
                error_rate > self.config.max_error_rate,
                f"row error rate {error_rate:.2%} exceeds allowed {self.config.max_error_rate:.2%}",
            ),
            (
                self.conflicting_ids > 0,
                f"{self.conflicting_ids} duplicate review IDs have conflicting content or organizations",
            ),
        )
        critical.extend(message for include, message in entries if include)
        return critical


def audit_yandex_reviews(
    path: Path,
    *,
    config: YandexReviewAuditConfig | None = None,
) -> YandexReviewAudit:
    """Audit one JSONL file without retaining review or author values in the result."""
    active_config = config or YandexReviewAuditConfig()
    reference_date = active_config.reference_date or datetime.now(UTC).date()
    format_result = detect_dataset_format(path, expected=DatasetFormat.JSONL)
    accumulator = _AuditAccumulator(active_config, reference_date)
    with path.open("rb") as source:
        for line_number, raw_line in enumerate(source, start=1):
            accumulator.observe(line_number, raw_line)
    return accumulator.build(path, format_result)


def _markdown(report: YandexReviewAudit) -> str:
    status = "usable" if report.is_usable else "blocked"
    critical = "\n".join(f"- {message}" for message in report.critical_errors) or "- None"
    warnings = "\n".join(f"- {message}" for message in report.warnings) or "- None"
    ratings = ", ".join(f"{rating}: {count}" for rating, count in report.rating_distribution.items()) or "none"
    counts = report.organization_review_counts
    organization_summary = (
        "none" if counts is None else f"min={counts.minimum}, p50={counts.p50}, p95={counts.p95}, max={counts.maximum}"
    )
    comparison_count = report.organizations_at_or_above_minimum_reviews
    return f"""# Yandex reviews dataset audit

- Status: **{status}**
- Source path: `{report.source_path}`
- Source SHA-256: `{report.source_sha256}`
- Size: {report.source_size_bytes} bytes
- Format: `{report.format.detected.value}` ({report.format.encoding or 'unknown encoding'})
- Lines: {report.lines_total}
- Contract-valid/invalid records: {report.contract_valid}/{report.contract_invalid}
- Row error rate: {report.error_rate:.2%}
- Unique review IDs: {report.unique_review_ids}
- Duplicate/conflicting review IDs: {report.duplicate_review_ids}/{report.conflicting_review_ids}
- Empty review texts: {report.empty_review_texts}
- Unique organizations: {report.unique_organizations}
- Organization review counts: {organization_summary}
- Organizations with at least {report.minimum_reviews_for_comparison} reviews: {comparison_count}
- Rating distribution: {ratings}
- Publication range: {report.published_at_min or 'unknown'} to {report.published_at_max or 'unknown'}
- Future-dated reviews against {report.reference_date}: {report.future_dates}
- Unique search queries: {report.unique_search_queries}

## Critical errors

{critical}

## Warnings

{warnings}

The report intentionally excludes review text and author values. Error samples contain only line numbers, field names,
types, hashes, and sanitized validation messages.
"""


def write_yandex_review_audit(
    report: YandexReviewAudit,
    report_dir: Path,
) -> tuple[Path, Path, Path]:
    """Write JSON, Markdown, and sanitized error artifacts."""
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "yandex-reviews-audit.json"
    markdown_path = report_dir / "yandex-reviews-audit.md"
    errors_path = report_dir / "yandex-reviews-errors.jsonl"
    json_path.write_text(f"{report.model_dump_json(indent=2)}\n", encoding="utf-8")
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    with errors_path.open("w", encoding="utf-8") as target:
        for error in report.errors:
            target.write(f"{error.model_dump_json()}\n")
    return json_path, markdown_path, errors_path
