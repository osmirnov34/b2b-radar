"""Read-only, fail-safe consumer of public stage-13 ML exports."""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class _PublicModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PublicTopic(_PublicModel):
    topic_id: int = Field(ge=0)
    name: str
    keywords: list[str]
    original_records: int = Field(ge=1)
    final_records: int = Field(ge=1)
    reassigned_records: int = Field(ge=0)
    original_languages: dict[str, int]
    original_unique_videos: int = Field(ge=0)
    mean_probability: float = Field(ge=0, le=1)


class PublicAssignment(_PublicModel):
    corpus_id: str = Field(pattern=r"^corpus:[0-9a-f]{64}$")
    topic_id: int | None = Field(default=None, ge=0)
    source: str = Field(pattern=r"^(hdbscan|reassignment|outlier)$")
    confidence: float = Field(ge=0, le=1)
    outlier_reason: str | None = None


class PublicQuality(_PublicModel):
    evaluation_status: str = Field(pattern=r"^(pass|pass_with_warnings|fail)$")
    preliminary: bool
    records: int = Field(ge=0)
    topics: int = Field(ge=0)
    final_outlier_share: float = Field(ge=0, le=1)
    mean_bootstrap_ari: float | None = Field(default=None, ge=-1, le=1)
    manual_annotations: int = Field(ge=0)
    warnings: list[str]


class _PublicExportConfig(_PublicModel):
    schema_version: int = 1
    include_research_text: bool
    require_final_evaluation: bool
    include_topic_keywords: bool
    maximum_keywords_per_topic: int


class _PublicManifest(_PublicModel):
    schema_version: int = 1
    config: _PublicExportConfig
    corpus_manifest_path: str
    corpus_manifest_sha256: str
    clustering_manifest_path: str
    clustering_manifest_sha256: str
    topic_manifest_path: str
    topic_manifest_sha256: str
    reassignment_manifest_path: str
    reassignment_manifest_sha256: str
    evaluation_manifest_path: str
    evaluation_manifest_sha256: str
    topics_path: str
    topics_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    assignments_path: str
    assignments_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    quality_path: str
    quality_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    research_assignments_path: str | None = None
    research_assignments_sha256: str | None = None
    records: int = Field(ge=0)
    topics: int = Field(ge=0)
    outliers: int = Field(ge=0)
    created_at: str


@dataclass(frozen=True)
class MLSnapshot:
    topics: tuple[PublicTopic, ...]
    assignments: tuple[PublicAssignment, ...]
    quality: PublicQuality
    assignments_by_topic: dict[int | None, tuple[PublicAssignment, ...]]


class SnapshotUnavailableError(RuntimeError):
    """No valid public snapshot has been loaded."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_path(root: Path, configured: str, expected_name: str) -> Path:
    path = Path(configured).resolve()
    if path.parent != root or path.name != expected_name:
        msg = f"public export artifact must be {expected_name!r} beside export-manifest.json"
        raise ValueError(msg)
    return path


def _load_jsonl(path: Path, model: type[_PublicModel]) -> list[_PublicModel]:
    items: list[_PublicModel] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                items.append(model.model_validate_json(line))
            except ValueError as exc:
                msg = f"invalid {path.name} record at line {line_number}: {exc}"
                raise ValueError(msg) from exc
    return items


def load_public_snapshot(manifest_path: Path, *, allow_unreliable: bool = False) -> MLSnapshot:
    """Load only the strict, aggregate public export contract."""
    manifest = _PublicManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    if manifest.config.include_research_text or manifest.research_assignments_path is not None:
        msg = "web application refuses manifests that reference sensitive research exports"
        raise ValueError(msg)
    root = manifest_path.resolve().parent
    topics_path = _artifact_path(root, manifest.topics_path, "topics.jsonl")
    assignments_path = _artifact_path(root, manifest.assignments_path, "assignments.jsonl")
    quality_path = _artifact_path(root, manifest.quality_path, "quality.json")
    for path, expected in (
        (topics_path, manifest.topics_sha256),
        (assignments_path, manifest.assignments_sha256),
        (quality_path, manifest.quality_sha256),
    ):
        if not path.is_file() or _sha256_file(path) != expected:
            msg = f"public export artifact is missing or has an invalid checksum: {path.name}"
            raise ValueError(msg)
    topics = tuple(PublicTopic.model_validate(item) for item in _load_jsonl(topics_path, PublicTopic))
    assignments = tuple(
        PublicAssignment.model_validate(item) for item in _load_jsonl(assignments_path, PublicAssignment)
    )
    quality = PublicQuality.model_validate_json(quality_path.read_text(encoding="utf-8"))
    if (quality.preliminary or quality.evaluation_status != "pass") and not allow_unreliable:
        msg = "ML snapshot is preliminary or did not pass evaluation"
        raise ValueError(msg)
    if len(topics) != manifest.topics or len(assignments) != manifest.records:
        msg = "public export counts do not match its manifest"
        raise ValueError(msg)
    topic_ids = {topic.topic_id for topic in topics}
    if topic_ids != set(range(manifest.topics)):
        msg = "public topics must use contiguous normalized IDs"
        raise ValueError(msg)
    grouped: dict[int | None, list[PublicAssignment]] = {topic_id: [] for topic_id in topic_ids}
    grouped[None] = []
    corpus_ids: set[str] = set()
    for assignment in assignments:
        if assignment.corpus_id in corpus_ids:
            msg = "public assignments contain duplicate corpus IDs"
            raise ValueError(msg)
        corpus_ids.add(assignment.corpus_id)
        if assignment.topic_id not in grouped:
            msg = "public assignment refers to an unknown topic"
            raise ValueError(msg)
        if (assignment.topic_id is None) != (assignment.source == "outlier"):
            msg = "public assignment has an ambiguous outlier/topic ownership"
            raise ValueError(msg)
        grouped[assignment.topic_id].append(assignment)
    if len(grouped[None]) != manifest.outliers:
        msg = "public outlier count does not match its manifest"
        raise ValueError(msg)
    frozen_groups = {key: tuple(value) for key, value in grouped.items()}
    return MLSnapshot(topics=topics, assignments=assignments, quality=quality, assignments_by_topic=frozen_groups)


class MLSnapshotRepository:
    """Thread-safe hot-reloading snapshot with last-known-good fallback."""

    def __init__(
        self,
        manifest_path: Path | None,
        *,
        enabled: bool,
        allow_unreliable: bool = False,
    ) -> None:
        self._manifest_path = manifest_path
        self._enabled = enabled
        self._allow_unreliable = allow_unreliable
        self._snapshot: MLSnapshot | None = None
        self._manifest_mtime_ns: int | None = None
        self._lock = threading.RLock()
        self.last_error: str | None = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def get_snapshot(self) -> MLSnapshot:
        with self._lock:
            if not self._enabled:
                msg = "ML results are disabled"
                raise SnapshotUnavailableError(msg)
            if self._manifest_path is None:
                msg = "ML export manifest is not configured"
                raise SnapshotUnavailableError(msg)
            try:
                mtime = self._manifest_path.stat().st_mtime_ns
                if self._snapshot is None or mtime != self._manifest_mtime_ns:
                    candidate = load_public_snapshot(
                        self._manifest_path,
                        allow_unreliable=self._allow_unreliable,
                    )
                    self._snapshot = candidate
                    self._manifest_mtime_ns = mtime
                    self.last_error = None
            except OSError as exc:
                self.last_error = "ML snapshot reload failed because a public artifact is unavailable"
                if self._snapshot is None:
                    raise SnapshotUnavailableError(self.last_error) from exc
            except ValueError as exc:
                self.last_error = "ML snapshot reload failed public-contract validation"
                if self._snapshot is None:
                    raise SnapshotUnavailableError(self.last_error) from exc
            if self._snapshot is None:  # pragma: no cover - defensive invariant
                msg = "ML snapshot is unavailable"
                raise SnapshotUnavailableError(msg)
            return self._snapshot

    def topic(self, topic_id: int) -> PublicTopic | None:
        snapshot = self.get_snapshot()
        return next((topic for topic in snapshot.topics if topic.topic_id == topic_id), None)
