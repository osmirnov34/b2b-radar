"""Deterministic, group-safe splitting of exported comments datasets."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, BinaryIO, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.analysis.cleaning import THANKS_TOKENS
from src.analysis.schemas import ExportedComment

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_RATIO_TOLERANCE = 1e-9
_MAX_ACKNOWLEDGEMENT_TOKENS = 3


class _HashUpdater(Protocol):
    def update(self, data: bytes, /) -> None: ...


class SplitName(StrEnum):
    DEVELOPMENT = "development"
    VALIDATION = "validation"
    TEST = "test"


class _SplitModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SplitConfig(_SplitModel):
    development_ratio: float = Field(default=0.70, gt=0, lt=1)
    validation_ratio: float = Field(default=0.15, gt=0, lt=1)
    test_ratio: float = Field(default=0.15, gt=0, lt=1)
    seed: int = 42
    min_informative_chars: int = Field(default=20, ge=1)
    min_informative_tokens: int = Field(default=4, ge=1)

    @model_validator(mode="after")
    def validate_ratios(self) -> SplitConfig:
        total = self.development_ratio + self.validation_ratio + self.test_ratio
        if abs(total - 1.0) > _RATIO_TOLERANCE:
            msg = f"split ratios must sum to 1.0, got {total}"
            raise ValueError(msg)
        return self


class SplitAssignment(_SplitModel):
    group_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    split: SplitName
    records: int = Field(ge=1)


class SplitStats(_SplitModel):
    input_records: int = Field(ge=0)
    input_groups: int = Field(ge=0)
    written_records: dict[SplitName, int]
    group_counts: dict[SplitName, int]
    removed_content_leaks: dict[SplitName, int]
    ignored_noise_overlaps: int = Field(ge=0)


class DatasetSplitManifest(_SplitModel):
    schema_version: int = 1
    source_path: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    config: SplitConfig
    assignments: list[SplitAssignment]
    stats: SplitStats
    output_sha256: dict[SplitName, str]
    created_at: datetime


def normalize_leakage_text(text: str) -> str:
    """Normalize only enough to identify exact textual leakage."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def is_informative_leakage_text(text: str, config: SplitConfig) -> bool:
    """Exclude acknowledgements and short generic phrases from leakage constraints."""
    normalized = normalize_leakage_text(text)
    tokens = _TOKEN_RE.findall(normalized)
    if len(normalized) < config.min_informative_chars or len(tokens) < config.min_informative_tokens:
        return False
    return not (
        len(tokens) <= _MAX_ACKNOWLEDGEMENT_TOKENS and all(token in THANKS_TOKENS for token in tokens)
    )


def _group_key(comment: ExportedComment, line_number: int) -> str:
    if comment.video_id.strip():
        return f"video_id:{comment.video_id.strip()}"
    if comment.video_url.strip():
        return f"video_url:{comment.video_url.strip()}"
    channel_title = f"{comment.video_channel.strip()}\x1f{comment.video_title.strip()}"
    if channel_title != "\x1f":
        return f"channel_title:{channel_title}"
    if comment.comment_id.strip():
        return f"comment_id:{comment.comment_id.strip()}"
    return f"line:{line_number}"


def _assign_group(group_key: str, config: SplitConfig) -> SplitName:
    digest = hashlib.sha256(f"{config.seed}\x1f{group_key}".encode()).digest()
    fraction = int.from_bytes(digest[:8]) / 2**64
    if fraction < config.development_ratio:
        return SplitName.DEVELOPMENT
    if fraction < config.development_ratio + config.validation_ratio:
        return SplitName.VALIDATION
    return SplitName.TEST


def _iter_comments(
    source: BinaryIO,
    *,
    source_hash: _HashUpdater | None = None,
) -> Iterator[tuple[int, bytes, ExportedComment]]:
    for line_number, raw_line in enumerate(source, start=1):
        if source_hash is not None:
            source_hash.update(raw_line)
        if not raw_line.strip():
            continue
        try:
            data: Any = json.loads(raw_line.decode("utf-8-sig"))
            comment = ExportedComment.model_validate(data)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            msg = f"invalid JSONL record at line {line_number}: {type(exc).__name__}"
            raise ValueError(msg) from exc
        yield line_number, raw_line, comment


def _text_digest(text: str) -> bytes:
    return hashlib.sha256(normalize_leakage_text(text).encode()).digest()


def _output_paths(output_dir: Path) -> dict[SplitName, Path]:
    return {name: output_dir / f"{name.value}.jsonl" for name in SplitName}


def _write_split_files(
    source_path: Path,
    paths: dict[SplitName, Path],
    assignments: dict[str, SplitName],
    owned_content: dict[SplitName, set[bytes]],
    config: SplitConfig,
) -> tuple[Counter[SplitName], Counter[SplitName], dict[SplitName, str]]:
    temporary = {name: path.with_suffix(f"{path.suffix}.tmp") for name, path in paths.items()}
    output_hashes = {name: hashlib.sha256() for name in SplitName}
    removed: Counter[SplitName] = Counter(dict.fromkeys(SplitName, 0))
    written: Counter[SplitName] = Counter(dict.fromkeys(SplitName, 0))
    handles: dict[SplitName, BinaryIO] = {}
    try:
        handles = {name: path.open("wb") for name, path in temporary.items()}
        with source_path.open("rb") as source:
            for line_number, raw_line, comment in _iter_comments(source):
                split = assignments[_group_key(comment, line_number)]
                if is_informative_leakage_text(comment.comment_text, config):
                    digest = _text_digest(comment.comment_text)
                    if digest not in owned_content[split]:
                        removed[split] += 1
                        continue
                handles[split].write(raw_line)
                output_hashes[split].update(raw_line)
                written[split] += 1
    except Exception:
        for handle in handles.values():
            handle.close()
        for path in temporary.values():
            path.unlink(missing_ok=True)
        raise
    else:
        for handle in handles.values():
            handle.close()
        for name, path in paths.items():
            temporary[name].replace(path)
    return removed, written, {name: digest.hexdigest() for name, digest in output_hashes.items()}


def split_comments_jsonl(
    source_path: Path,
    output_dir: Path,
    *,
    config: SplitConfig | None = None,
    force: bool = False,
) -> DatasetSplitManifest:
    """Split JSONL by source group and remove informative cross-split exact duplicates."""
    active_config = config or SplitConfig()
    paths = _output_paths(output_dir)
    manifest_path = output_dir / "split-manifest.json"
    existing = [path for path in (*paths.values(), manifest_path) if path.exists()]
    if existing and not force:
        msg = f"refusing to overwrite existing split output: {existing[0]}"
        raise FileExistsError(msg)

    source_hash = hashlib.sha256()
    group_sizes: Counter[str] = Counter()
    with source_path.open("rb") as source:
        for line_number, _, comment in _iter_comments(source, source_hash=source_hash):
            group_sizes[_group_key(comment, line_number)] += 1

    assignments = {group: _assign_group(group, active_config) for group in group_sizes}
    content_hashes: dict[SplitName, set[bytes]] = {name: set() for name in SplitName}
    noise_splits: dict[bytes, set[SplitName]] = {}
    with source_path.open("rb") as source:
        for line_number, _, comment in _iter_comments(source):
            split = assignments[_group_key(comment, line_number)]
            digest = _text_digest(comment.comment_text)
            if is_informative_leakage_text(comment.comment_text, active_config):
                content_hashes[split].add(digest)
            else:
                noise_splits.setdefault(digest, set()).add(split)

    owned_content = {
        SplitName.DEVELOPMENT: content_hashes[SplitName.DEVELOPMENT],
        SplitName.VALIDATION: content_hashes[SplitName.VALIDATION] - content_hashes[SplitName.DEVELOPMENT],
        SplitName.TEST: content_hashes[SplitName.TEST]
        - content_hashes[SplitName.DEVELOPMENT]
        - content_hashes[SplitName.VALIDATION],
    }
    group_counts = Counter(assignments.values())
    ignored_noise_overlaps = sum(1 for splits in noise_splits.values() if len(splits) > 1)

    output_dir.mkdir(parents=True, exist_ok=True)
    removed, written, output_hashes = _write_split_files(
        source_path,
        paths,
        assignments,
        owned_content,
        active_config,
    )

    public_assignments = sorted(
        (
            SplitAssignment(
                group_sha256=hashlib.sha256(group.encode()).hexdigest(),
                split=assignments[group],
                records=records,
            )
            for group, records in group_sizes.items()
        ),
        key=lambda item: item.group_sha256,
    )
    manifest = DatasetSplitManifest(
        source_path=str(source_path),
        source_sha256=source_hash.hexdigest(),
        config=active_config,
        assignments=public_assignments,
        stats=SplitStats(
            input_records=sum(group_sizes.values()),
            input_groups=len(group_sizes),
            written_records=dict(written),
            group_counts=dict(group_counts),
            removed_content_leaks=dict(removed),
            ignored_noise_overlaps=ignored_noise_overlaps,
        ),
        output_sha256=output_hashes,
        created_at=datetime.now(UTC),
    )
    manifest_path.write_text(f"{manifest.model_dump_json(indent=2)}\n", encoding="utf-8")
    return manifest
