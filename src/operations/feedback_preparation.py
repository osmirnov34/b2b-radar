"""Prepare canonical analytical feedback artifacts from verified sources."""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from src.analytics import FeedbackPreparationManifest, FeedbackSource
from src.sources import iter_yandex_reviews, yandex_review_to_feedback

if TYPE_CHECKING:
    from typing import IO

    from src.sources import YandexReview


class FeedbackPreparationError(RuntimeError):
    """Raised when a canonical artifact cannot be created safely."""


class _Digest(Protocol):
    def update(self, value: bytes) -> None: ...

    def hexdigest(self) -> str: ...


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _temporary_file(target: Path) -> tuple[IO[bytes], Path]:
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(  # noqa: SIM115 -- ownership is transferred to the caller.
        mode="w+b",
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
        delete=False,
    )
    return handle, Path(handle.name)


def _require_available(path: Path) -> None:
    if path.exists():
        msg = f"refusing to replace existing artifact: {path}"
        raise FeedbackPreparationError(msg)


def _publish_new(temporary_path: Path, target_path: Path) -> None:
    """Publish without a check/use race and never replace an existing artifact."""
    try:
        os.link(temporary_path, target_path)
    except FileExistsError as exc:
        msg = f"refusing to replace existing artifact: {target_path}"
        raise FeedbackPreparationError(msg) from exc
    temporary_path.unlink()


@dataclass
class _PreparationAccumulator:
    feedback_ids: set[str] = field(default_factory=set)
    organization_ids: set[str] = field(default_factory=set)
    ratings: Counter[int] = field(default_factory=Counter)
    published_min: date | None = None
    published_max: date | None = None
    output_digest: _Digest = field(default_factory=hashlib.sha256)

    def encode(self, review: YandexReview) -> bytes:
        """Validate uniqueness, update aggregates, and encode one source record."""
        feedback = yandex_review_to_feedback(review)
        if feedback.feedback_id in self.feedback_ids:
            msg = "duplicate feedback identifier detected"
            raise FeedbackPreparationError(msg)
        self.feedback_ids.add(feedback.feedback_id)
        self.organization_ids.add(feedback.organization_id)
        self.ratings[feedback.rating] += 1
        self.published_min = (
            min(self.published_min, feedback.published_on) if self.published_min else feedback.published_on
        )
        self.published_max = (
            max(self.published_max, feedback.published_on) if self.published_max else feedback.published_on
        )
        encoded = f"{feedback.model_dump_json()}\n".encode()
        self.output_digest.update(encoded)
        return encoded


def _write_canonical_jsonl(source_path: Path, output_handle: IO[bytes]) -> _PreparationAccumulator:
    accumulator = _PreparationAccumulator()
    output_handle.writelines(accumulator.encode(review) for review in iter_yandex_reviews(source_path))
    output_handle.flush()
    os.fsync(output_handle.fileno())
    return accumulator


def prepare_yandex_feedback(
    source_path: Path,
    output_path: Path,
    manifest_path: Path,
) -> FeedbackPreparationManifest:
    """Stream Yandex reviews and safely publish a new canonical artifact pair."""
    if not source_path.is_file():
        msg = f"source file not found: {source_path}"
        raise FeedbackPreparationError(msg)
    if output_path.resolve() == manifest_path.resolve():
        msg = "output and manifest paths must differ"
        raise FeedbackPreparationError(msg)
    _require_available(output_path)
    _require_available(manifest_path)

    output_handle, temporary_output = _temporary_file(output_path)
    temporary_manifest: Path | None = None
    output_published = False
    try:
        with output_handle:
            accumulator = _write_canonical_jsonl(source_path, output_handle)

        manifest = FeedbackPreparationManifest(
            source=FeedbackSource.YANDEX_MAPS,
            source_path=str(source_path),
            source_size_bytes=source_path.stat().st_size,
            source_sha256=_sha256(source_path),
            output_path=str(output_path),
            output_size_bytes=temporary_output.stat().st_size,
            output_sha256=accumulator.output_digest.hexdigest(),
            prepared_at=datetime.now(UTC),
            records=len(accumulator.feedback_ids),
            unique_organizations=len(accumulator.organization_ids),
            rating_distribution=dict(sorted(accumulator.ratings.items())),
            published_on_min=accumulator.published_min,
            published_on_max=accumulator.published_max,
        )

        manifest_handle, temporary_manifest = _temporary_file(manifest_path)
        with manifest_handle:
            manifest_handle.write(f"{manifest.model_dump_json(indent=2)}\n".encode())
            manifest_handle.flush()
            os.fsync(manifest_handle.fileno())

        _publish_new(temporary_output, output_path)
        output_published = True
        _publish_new(temporary_manifest, manifest_path)
    except BaseException:
        if output_published:
            output_path.unlink(missing_ok=True)
        raise
    else:
        return manifest
    finally:
        temporary_output.unlink(missing_ok=True)
        if temporary_manifest is not None:
            temporary_manifest.unlink(missing_ok=True)
