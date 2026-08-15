import hashlib
import heapq
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.analysis.inspection import DatasetInspection
from src.analysis.schemas import ExportedComment


class SampleMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    source_path: str
    source_sha256: str
    output_path: str
    output_sha256: str
    requested_records: int = Field(ge=1)
    written_records: int = Field(ge=0)
    random_seed: int
    max_records_per_video: int = Field(ge=1)
    created_at: datetime


def write_inspection_reports(report: DatasetInspection, report_dir: Path) -> tuple[Path, Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "dataset-profile.json"
    markdown_path = report_dir / "dataset-profile.md"
    errors_path = report_dir / "inspection-errors.jsonl"

    json_path.write_text(f"{report.model_dump_json(indent=2)}\n", encoding="utf-8")
    markdown_path.write_text(_inspection_markdown(report), encoding="utf-8")
    with errors_path.open("w", encoding="utf-8") as target:
        for error in report.errors:
            target.write(f"{error.model_dump_json()}\n")
    return json_path, markdown_path, errors_path


def _inspection_markdown(report: DatasetInspection) -> str:
    status = "usable" if report.is_usable else "blocked"
    critical = "\n".join(f"- {message}" for message in report.critical_errors) or "- None"
    lengths = report.text_lengths
    length_summary = (
        f"min={lengths.minimum}, p50={lengths.p50}, p95={lengths.p95}, max={lengths.maximum}"
        if lengths is not None
        else "no non-empty text"
    )
    return f"""# Dataset inspection

- Status: **{status}**
- Path: `{report.path}`
- SHA-256: `{report.sha256}`
- Size: {report.size_bytes} bytes
- Expected format: `{report.format.expected}`
- Detected format: `{report.format.detected}`
- Encoding: `{report.format.encoding or 'unknown'}`
- Lines: {report.lines_total}
- Contract-valid records: {report.contract_valid}
- Row error rate: {report.error_rate:.2%}
- Non-empty comments: {report.non_empty_text}
- Unique authors: {report.unique_authors}
- Unique videos: {report.unique_videos}
- Unique channels: {report.unique_channels}
- Unique queries: {report.unique_queries}
- Duplicate texts/groups: {report.duplicate_texts}/{report.duplicate_text_groups}
- Text lengths: {length_summary}

## Critical errors

{critical}

Error samples are stored separately and contain only field names, types, hashes, and sanitized messages.
"""


def _sample_rank(seed: int, line_number: int, comment: ExportedComment) -> int:
    identity = "\x1f".join(
        (
            str(seed),
            str(line_number),
            comment.video_id,
            comment.comment_id,
            " ".join(comment.comment_text.casefold().split()),
        ),
    )
    return int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8])


def create_research_sample(
    source_path: Path,
    output_path: Path,
    *,
    sample_size: int,
    seed: int = 42,
    max_records_per_video: int = 100,
) -> SampleMetadata:
    if sample_size < 1:
        msg = "sample_size must be positive"
        raise ValueError(msg)
    if max_records_per_video < 1:
        msg = "max_records_per_video must be positive"
        raise ValueError(msg)

    source_hash = hashlib.sha256()
    candidates: dict[str, list[tuple[int, int, dict[str, Any]]]] = {}
    with source_path.open("rb") as source:
        for line_number, raw_line in enumerate(source, start=1):
            source_hash.update(raw_line)
            if not raw_line.strip():
                continue
            try:
                data: Any = json.loads(raw_line.decode("utf-8-sig"))
                comment = ExportedComment.model_validate(data)
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                continue
            if not comment.comment_text.strip() or not isinstance(data, dict):
                continue
            group = comment.video_id or f"__missing_video__:{line_number}"
            rank = _sample_rank(seed, line_number, comment)
            heap = candidates.setdefault(group, [])
            item = (-rank, line_number, data)
            if len(heap) < max_records_per_video:
                heapq.heappush(heap, item)
            elif rank < -heap[0][0]:
                heapq.heapreplace(heap, item)

    ranked_candidates = (
        (-negative_rank, line_number, data)
        for heap in candidates.values()
        for negative_rank, line_number, data in heap
    )
    selected = sorted(
        ranked_candidates,
        key=lambda item: (item[0], item[1]),
    )[:sample_size]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_hash = hashlib.sha256()
    with output_path.open("wb") as target:
        for _, _, data in selected:
            encoded = f"{json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n".encode()
            target.write(encoded)
            output_hash.update(encoded)

    metadata = SampleMetadata(
        source_path=str(source_path),
        source_sha256=source_hash.hexdigest(),
        output_path=str(output_path),
        output_sha256=output_hash.hexdigest(),
        requested_records=sample_size,
        written_records=len(selected),
        random_seed=seed,
        max_records_per_video=max_records_per_video,
        created_at=datetime.now(UTC),
    )
    metadata_path = output_path.with_suffix(f"{output_path.suffix}.meta.json")
    metadata_path.write_text(f"{metadata.model_dump_json(indent=2)}\n", encoding="utf-8")
    return metadata
