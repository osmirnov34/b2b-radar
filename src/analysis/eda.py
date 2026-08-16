"""Privacy-safe exploratory profiling for the development dataset."""

from __future__ import annotations

import hashlib
import heapq
import json
import re
from collections import Counter
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.analysis.cleaning import classify
from src.analysis.models import CleaningReason
from src.analysis.schemas import ExportedComment
from src.analysis.splitting import DatasetSplitManifest, SplitName, normalize_leakage_text
from src.infrastructure.extractor.language import detect_language
from src.infrastructure.extractor.noise import is_noise

if TYPE_CHECKING:
    from pathlib import Path

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
_REPEATED_CHAR_RE = re.compile(r"(.)\1{5,}", re.DOTALL)
_LENGTH_BUCKETS = (20, 50, 100, 200, 500, 1000)
_DUPLICATE_BUCKETS = (2, 3, 5, 10, 50, 100)
_MOSTLY_UPPERCASE_RATIO = 0.8


class _EDAModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EDAConfig(_EDAModel):
    language_sample_size: int = Field(default=20_000, ge=0)
    language_seed: int = 42
    top_groups: int = Field(default=20, ge=0, le=1000)


class NumericSummary(_EDAModel):
    minimum: int = Field(ge=0)
    p50: int = Field(ge=0)
    p75: int = Field(ge=0)
    p90: int = Field(ge=0)
    p95: int = Field(ge=0)
    p99: int = Field(ge=0)
    maximum: int = Field(ge=0)
    mean: float = Field(ge=0)


class HashedGroupCount(_EDAModel):
    group_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    records: int = Field(ge=1)


class EDAProfile(_EDAModel):
    schema_version: int = 1
    split: SplitName = SplitName.DEVELOPMENT
    dataset_path: str
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_path: str
    config: EDAConfig
    records: int = Field(ge=0)
    unique_texts: int = Field(ge=0)
    duplicate_texts: int = Field(ge=0)
    duplicate_groups: int = Field(ge=0)
    unique_authors: int = Field(ge=0)
    unique_videos: int = Field(ge=0)
    unique_channels: int = Field(ge=0)
    unique_queries: int = Field(ge=0)
    missing_fields: dict[str, int]
    character_lengths: NumericSummary | None = None
    token_lengths: NumericSummary | None = None
    character_length_buckets: dict[str, int]
    duplicate_group_size_buckets: dict[str, int]
    noise_categories: dict[str, int]
    languages: dict[str, int]
    language_sample_records: int = Field(ge=0)
    monthly_records: dict[str, int]
    top_video_groups: list[HashedGroupCount]
    top_channel_groups: list[HashedGroupCount]
    top_query_groups: list[HashedGroupCount]
    created_at: datetime

    @model_validator(mode="after")
    def validate_counts(self) -> EDAProfile:
        if self.unique_texts + self.duplicate_texts != self.records:
            msg = "unique_texts and duplicate_texts must add up to records"
            raise ValueError(msg)
        if sum(self.languages.values()) != self.language_sample_records:
            msg = "language counts must add up to language_sample_records"
            raise ValueError(msg)
        return self


def _quantile(sorted_values: list[int], fraction: float) -> int:
    if not sorted_values:
        return 0
    index = round((len(sorted_values) - 1) * fraction)
    return sorted_values[index]


def summarize_numbers(values: list[int]) -> NumericSummary | None:
    if not values:
        return None
    ordered = sorted(values)
    return NumericSummary(
        minimum=ordered[0],
        p50=_quantile(ordered, 0.50),
        p75=_quantile(ordered, 0.75),
        p90=_quantile(ordered, 0.90),
        p95=_quantile(ordered, 0.95),
        p99=_quantile(ordered, 0.99),
        maximum=ordered[-1],
        mean=sum(ordered) / len(ordered),
    )


def _bucket(value: int, boundaries: tuple[int, ...]) -> str:
    for boundary in boundaries:
        if value <= boundary:
            return f"<= {boundary}"
    return f"> {boundaries[-1]}"


def _hashed_top(counter: Counter[str], limit: int) -> list[HashedGroupCount]:
    return [
        HashedGroupCount(group_sha256=hashlib.sha256(key.encode()).hexdigest(), records=count)
        for key, count in counter.most_common(limit)
    ]


def _language_rank(seed: int, line_number: int, text: str) -> int:
    key = f"{seed}\x1f{line_number}\x1f{normalize_leakage_text(text)}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8])


def _add_language_candidate(
    heap: list[tuple[int, str]],
    *,
    text: str,
    rank: int,
    limit: int,
) -> None:
    if limit == 0:
        return
    item = (-rank, text)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif rank < -heap[0][0]:
        heapq.heapreplace(heap, item)


def _load_development_manifest(manifest_path: Path, dataset_path: Path) -> DatasetSplitManifest:
    manifest = DatasetSplitManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    expected = manifest.output_sha256.get(SplitName.DEVELOPMENT)
    digest = hashlib.sha256()
    with dataset_path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        msg = "dataset checksum does not match development output in split manifest"
        raise ValueError(msg)
    return manifest


def profile_development_dataset(
    dataset_path: Path,
    manifest_path: Path,
    *,
    config: EDAConfig | None = None,
) -> EDAProfile:
    """Profile only the checksum-verified development split without persisting raw values."""
    active_config = config or EDAConfig()
    manifest = _load_development_manifest(manifest_path, dataset_path)
    expected_records = manifest.stats.written_records.get(SplitName.DEVELOPMENT, 0)
    text_counts: Counter[str] = Counter()
    authors: set[str] = set()
    videos: Counter[str] = Counter()
    channels: Counter[str] = Counter()
    queries: Counter[str] = Counter()
    missing: Counter[str] = Counter()
    noise: Counter[str] = Counter()
    months: Counter[str] = Counter()
    char_lengths: list[int] = []
    token_lengths: list[int] = []
    char_buckets: Counter[str] = Counter()
    language_heap: list[tuple[int, str]] = []
    records = 0
    profile_hash = hashlib.sha256()

    with dataset_path.open("rb") as source:
        for line_number, raw_line in enumerate(source, start=1):
            profile_hash.update(raw_line)
            if not raw_line.strip():
                continue
            try:
                data: Any = json.loads(raw_line.decode("utf-8-sig"))
                comment = ExportedComment.model_validate(data)
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                msg = f"invalid development record at line {line_number}: {type(exc).__name__}"
                raise ValueError(msg) from exc
            records += 1
            text = comment.comment_text.strip()
            normalized = normalize_leakage_text(text)
            tokens = _TOKEN_RE.findall(normalized)
            char_lengths.append(len(text))
            token_lengths.append(len(tokens))
            char_buckets[_bucket(len(text), _LENGTH_BUCKETS)] += 1
            text_counts[normalized] += 1
            _add_language_candidate(
                language_heap,
                text=text,
                rank=_language_rank(active_config.language_seed, line_number, text),
                limit=active_config.language_sample_size,
            )

            if not text:
                noise["empty"] += 1
            decision = classify(text, min_length=0)
            if decision.reason == CleaningReason.ACKNOWLEDGEMENT:
                noise["acknowledgement"] += 1
            if is_noise(text):
                noise["pipeline_noise"] += 1
            if _URL_RE.search(text):
                noise["contains_url"] += 1
            if _REPEATED_CHAR_RE.search(text):
                noise["repeated_characters"] += 1
            letters = [char for char in text if char.isalpha()]
            if letters and sum(char.isupper() for char in letters) / len(letters) > _MOSTLY_UPPERCASE_RATIO:
                noise["mostly_uppercase"] += 1

            for field, value in (
                ("comment_author", comment.comment_author),
                ("video_id", comment.video_id),
                ("video_channel", comment.video_channel),
                ("search_query", comment.search_query),
                ("comment_published_at", comment.comment_published_at),
            ):
                if value is None or (isinstance(value, str) and not value.strip()):
                    missing[field] += 1
            if comment.comment_author:
                authors.add(comment.comment_author)
            if comment.video_id:
                videos[comment.video_id] += 1
            if comment.video_channel:
                channels[comment.video_channel] += 1
            if comment.search_query:
                queries[comment.search_query] += 1
            if comment.comment_published_at is not None:
                months[comment.comment_published_at.strftime("%Y-%m")] += 1

    if records != expected_records:
        msg = f"development record count {records} does not match manifest count {expected_records}"
        raise ValueError(msg)
    if profile_hash.hexdigest() != manifest.output_sha256[SplitName.DEVELOPMENT]:
        msg = "development dataset changed while it was being profiled"
        raise ValueError(msg)

    duplicate_counts = [count for count in text_counts.values() if count > 1]
    duplicate_buckets = Counter(_bucket(count, _DUPLICATE_BUCKETS) for count in duplicate_counts)
    languages = Counter(detect_language(text) for _, text in language_heap)
    dataset_sha256 = manifest.output_sha256[SplitName.DEVELOPMENT]
    return EDAProfile(
        dataset_path=str(dataset_path),
        dataset_sha256=dataset_sha256,
        manifest_path=str(manifest_path),
        config=active_config,
        records=records,
        unique_texts=len(text_counts),
        duplicate_texts=records - len(text_counts),
        duplicate_groups=len(duplicate_counts),
        unique_authors=len(authors),
        unique_videos=len(videos),
        unique_channels=len(channels),
        unique_queries=len(queries),
        missing_fields=dict(missing),
        character_lengths=summarize_numbers(char_lengths),
        token_lengths=summarize_numbers(token_lengths),
        character_length_buckets=dict(char_buckets),
        duplicate_group_size_buckets=dict(duplicate_buckets),
        noise_categories=dict(noise),
        languages=dict(languages),
        language_sample_records=len(language_heap),
        monthly_records=dict(sorted(months.items())),
        top_video_groups=_hashed_top(videos, active_config.top_groups),
        top_channel_groups=_hashed_top(channels, active_config.top_groups),
        top_query_groups=_hashed_top(queries, active_config.top_groups),
        created_at=datetime.now(UTC),
    )
