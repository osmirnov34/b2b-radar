"""Read-only, checksum-bound loading and reporting for completed ML artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from src.ml.cleaning_dataset import DatasetCleaningManifest
from src.ml.clustering import ClusteringManifest
from src.ml.corpus import CorpusManifest
from src.ml.inspection import DatasetInspection
from src.ml.outlier_reassignment import OutlierReassignmentManifest
from src.ml.semantic_deduplication import SemanticDeduplicationManifest
from src.ml.topic_representation import TopicRepresentation, TopicRepresentationManifest

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

_MATRIX_DIMENSIONS = 2
_MINIMUM_PLOT_DIMENSIONS = 2


class _ReportModel(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class AnalysisSummary(_ReportModel):
    records: int = Field(ge=0)
    topics: int = Field(ge=0)
    outliers: int = Field(ge=0)
    outlier_share: float = Field(ge=0, le=1)
    mean_confidence: float = Field(ge=0, le=1)
    largest_topic_share: float = Field(ge=0, le=1)
    pipeline_status: str
    warnings: list[str]


class TopicSummaryRow(_ReportModel):
    topic_id: int = Field(ge=0)
    name: str
    records: int = Field(ge=1)
    corpus_share: float = Field(gt=0, le=1)
    mean_probability: float = Field(ge=0, le=1)
    unique_videos: int = Field(ge=0)
    keywords: list[str]


class ProcessingFlowStep(_ReportModel):
    stage: str
    records: int = Field(ge=0)
    note: str = ""


class ReportManifest(_ReportModel):
    report_schema_version: int = 1
    source_pipeline_schema_version: int = Field(ge=1)
    run_id: str
    pipeline_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    private_text_included: bool = False
    generated_at: datetime
    files: list[str]


@dataclass(frozen=True)
class AnalysisArtifacts:
    """Verified arrays and aggregate models; raw corpus text is deliberately not loaded."""

    run_dir: Path
    run_id: str
    pipeline_schema_version: int
    pipeline_manifest_sha256: str
    corpus_path: Path
    coordinates: Any
    labels: Any
    confidence: Any
    corpus: CorpusManifest
    clustering: ClusteringManifest
    topics_manifest: TopicRepresentationManifest
    topics: tuple[TopicRepresentation, ...]
    reassignment: OutlierReassignmentManifest | None
    summary: AnalysisSummary


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_path(run_dir: Path, raw_path: str, expected_sha256: str) -> Path:
    path = Path(raw_path).resolve()
    if not path.is_relative_to(run_dir):
        msg = f"report artifact escapes the selected run directory: {path.name}"
        raise ValueError(msg)
    if not path.is_file():
        msg = f"report artifact is missing: {path.name}"
        raise FileNotFoundError(msg)
    if _sha256_file(path) != expected_sha256:
        msg = f"report artifact checksum mismatch: {path.name}"
        raise ValueError(msg)
    return path


def _load_topics(path: Path) -> tuple[TopicRepresentation, ...]:
    with path.open(encoding="utf-8") as source:
        topics = tuple(TopicRepresentation.model_validate_json(line) for line in source if line.strip())
    if [topic.topic_id for topic in topics] != list(range(len(topics))):
        msg = "topic representations must be ordered with contiguous IDs"
        raise ValueError(msg)
    return topics


def load_analysis_artifacts(run_dir: Path) -> AnalysisArtifacts:
    """Load only verified aggregate artifacts from an existing run without modifying it."""
    import numpy as np

    resolved = run_dir.resolve()
    pipeline_path = resolved / "run-manifest.json"
    pipeline_data = json.loads(pipeline_path.read_text(encoding="utf-8"))
    run_id = str(pipeline_data["run_id"])
    pipeline_schema = int(pipeline_data.get("schema_version", 1))
    pipeline_status = str(pipeline_data["status"])

    corpus_path = resolved / "07-corpus/corpus-manifest.json"
    clustering_path = resolved / "09-clustering/clustering-manifest.json"
    topics_path = resolved / "10-topics/topic-representation-manifest.json"
    corpus = CorpusManifest.model_validate_json(corpus_path.read_text(encoding="utf-8"))
    clustering = ClusteringManifest.model_validate_json(clustering_path.read_text(encoding="utf-8"))
    topics_manifest = TopicRepresentationManifest.model_validate_json(topics_path.read_text(encoding="utf-8"))

    verified_corpus_path = _verified_path(resolved, corpus.corpus_path, corpus.corpus_sha256)
    coordinates_path = _verified_path(resolved, clustering.reduced_path, clustering.reduced_sha256)
    labels_path = _verified_path(resolved, clustering.labels_path, clustering.labels_sha256)
    confidence_path = _verified_path(resolved, clustering.probabilities_path, clustering.probabilities_sha256)
    representation_path = _verified_path(
        resolved,
        topics_manifest.representations_path,
        topics_manifest.representations_sha256,
    )
    reassignment_path = resolved / "11-reassignment/outlier-reassignment-manifest.json"
    reassignment = None
    if reassignment_path.is_file():
        reassignment = OutlierReassignmentManifest.model_validate_json(reassignment_path.read_text(encoding="utf-8"))
        labels_path = _verified_path(resolved, reassignment.final_labels_path, reassignment.final_labels_sha256)
        confidence_path = _verified_path(
            resolved,
            reassignment.final_confidence_path,
            reassignment.final_confidence_sha256,
        )

    coordinates = np.load(coordinates_path, mmap_mode="r", allow_pickle=False)
    labels = np.load(labels_path, mmap_mode="r", allow_pickle=False)
    confidence = np.load(confidence_path, mmap_mode="r", allow_pickle=False)
    if coordinates.ndim != _MATRIX_DIMENSIONS or coordinates.shape[1] < _MINIMUM_PLOT_DIMENSIONS:
        msg = "reduced coordinates must be a matrix with at least two columns"
        raise ValueError(msg)
    if coordinates.shape[0] != labels.shape[0] or labels.shape != confidence.shape:
        msg = "coordinates, labels, and confidence arrays are not row-aligned"
        raise ValueError(msg)
    if not np.isfinite(coordinates).all() or not np.isfinite(confidence).all():
        msg = "report arrays contain NaN or infinite values"
        raise ValueError(msg)
    topics = _load_topics(representation_path)
    records = int(labels.shape[0])
    outliers = int(np.count_nonzero(labels == -1))
    regular = labels[labels >= 0]
    counts = np.bincount(regular) if regular.size else np.asarray([], dtype=np.int64)
    largest_share = float(counts.max() / records) if counts.size and records else 0.0
    warnings = list(dict.fromkeys([*clustering.warnings, *(reassignment.warnings if reassignment else [])]))
    summary = AnalysisSummary(
        records=records,
        topics=len(topics),
        outliers=outliers,
        outlier_share=outliers / records if records else 0.0,
        mean_confidence=float(confidence.mean()) if records else 0.0,
        largest_topic_share=largest_share,
        pipeline_status=pipeline_status,
        warnings=warnings,
    )
    return AnalysisArtifacts(
        run_dir=resolved,
        run_id=run_id,
        pipeline_schema_version=pipeline_schema,
        pipeline_manifest_sha256=_sha256_file(pipeline_path),
        corpus_path=verified_corpus_path,
        coordinates=coordinates,
        labels=labels,
        confidence=confidence,
        corpus=corpus,
        clustering=clustering,
        topics_manifest=topics_manifest,
        topics=topics,
        reassignment=reassignment,
        summary=summary,
    )


def topic_summary_rows(artifacts: AnalysisArtifacts) -> list[TopicSummaryRow]:
    return [
        TopicSummaryRow(
            topic_id=topic.topic_id,
            name=topic.name,
            records=topic.records,
            corpus_share=topic.records / artifacts.summary.records,
            mean_probability=topic.mean_probability,
            unique_videos=topic.unique_videos,
            keywords=[keyword.term for keyword in topic.keywords],
        )
        for topic in artifacts.topics
    ]


def processing_flow(artifacts: AnalysisArtifacts) -> list[ProcessingFlowStep]:
    """Build aggregate record-flow steps from persisted stage manifests."""
    inspection = DatasetInspection.model_validate_json(
        (artifacts.run_dir / "01-inspection/dataset-profile.json").read_text(encoding="utf-8"),
    )
    cleaning = DatasetCleaningManifest.model_validate_json(
        (artifacts.run_dir / "04-cleaning/cleaning-manifest.json").read_text(encoding="utf-8"),
    )
    deduplication = SemanticDeduplicationManifest.model_validate_json(
        (artifacts.run_dir / "06-deduplication/semantic-deduplication-manifest.json").read_text(encoding="utf-8"),
    )
    return [
        ProcessingFlowStep(stage="valid parent comments", records=inspection.contract_valid),
        ProcessingFlowStep(
            stage="flattened text units",
            records=cleaning.stats.input_text_units,
            note="includes nested replies",
        ),
        ProcessingFlowStep(stage="after cleaning", records=cleaning.stats.output_text_units),
        ProcessingFlowStep(stage="after semantic deduplication", records=deduplication.result.n_kept),
        ProcessingFlowStep(stage="final corpus", records=artifacts.corpus.stats.output_records),
        ProcessingFlowStep(stage="assigned to topics", records=artifacts.summary.records - artifacts.summary.outliers),
        ProcessingFlowStep(stage="remaining outliers", records=artifacts.summary.outliers),
    ]


def stratified_plot_indices(
    labels: NDArray[np.integer[Any]],
    *,
    maximum: int = 20_000,
    seed: int = 42,
) -> NDArray[np.int64]:
    """Return deterministic plot indices while retaining every represented label."""
    import numpy as np

    values = np.asarray(labels)
    if values.ndim != 1:
        msg = "labels must be one-dimensional"
        raise ValueError(msg)
    if maximum < len(np.unique(values)):
        msg = "maximum plot points must cover every label"
        raise ValueError(msg)
    if len(values) <= maximum:
        return np.arange(len(values), dtype=np.int64)
    rng = np.random.default_rng(seed)
    selected_indices: list[int] = []
    for label in np.unique(values):
        candidates = np.flatnonzero(values == label)
        allocation = max(1, round(maximum * len(candidates) / len(values)))
        chosen_candidates = np.asarray(
            rng.choice(candidates, min(allocation, len(candidates)), replace=False),
            dtype=np.int64,
        )
        selected_indices.extend(int(index) for index in chosen_candidates)
    if len(selected_indices) > maximum:
        mandatory = [int(np.flatnonzero(values == label)[0]) for label in np.unique(values)]
        remaining_indices = sorted(set(selected_indices).difference(mandatory))
        chosen_remainder = np.asarray(
            rng.choice(remaining_indices, maximum - len(mandatory), replace=False),
            dtype=np.int64,
        )
        selected_indices = mandatory + [int(index) for index in chosen_remainder]
    elif len(selected_indices) < maximum:
        remaining_array = np.setdiff1d(np.arange(len(values)), np.asarray(selected_indices), assume_unique=False)
        chosen_additions = np.asarray(
            rng.choice(remaining_array, maximum - len(selected_indices), replace=False),
            dtype=np.int64,
        )
        selected_indices.extend(int(index) for index in chosen_additions)
    return np.asarray(sorted(selected_indices), dtype=np.int64)


def write_analysis_tables(
    artifacts: AnalysisArtifacts,
    output_root: Path,
    *,
    overwrite: bool = False,
) -> ReportManifest:
    """Write aggregate reports outside the immutable run directory."""
    target = (output_root / artifacts.run_id).resolve()
    if target.is_relative_to(artifacts.run_dir):
        msg = "analysis reports must be stored outside the ML run directory"
        raise ValueError(msg)
    if target.exists() and any(target.iterdir()) and not overwrite:
        msg = f"analysis report already exists: {target}"
        raise FileExistsError(msg)
    tables = target / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    summary_path = target / "summary.json"
    topics_path = tables / "topic-summary.csv"
    summary_path.write_text(f"{artifacts.summary.model_dump_json(indent=2)}\n", encoding="utf-8")
    rows = topic_summary_rows(artifacts)
    with topics_path.open("w", encoding="utf-8", newline="") as target_file:
        writer = csv.DictWriter(
            target_file,
            fieldnames=[
                "topic_id",
                "name",
                "records",
                "corpus_share",
                "mean_probability",
                "unique_videos",
                "keywords",
            ],
        )
        writer.writeheader()
        for row in rows:
            payload = row.model_dump()
            payload["keywords"] = " | ".join(row.keywords)
            writer.writerow(payload)
    manifest = ReportManifest(
        source_pipeline_schema_version=artifacts.pipeline_schema_version,
        run_id=artifacts.run_id,
        pipeline_manifest_sha256=artifacts.pipeline_manifest_sha256,
        generated_at=datetime.now(UTC),
        files=[str(summary_path), str(topics_path)],
    )
    manifest_path = target / "report-manifest.json"
    manifest_path.write_text(f"{manifest.model_dump_json(indent=2)}\n", encoding="utf-8")
    return manifest.model_copy(update={"files": [*manifest.files, str(manifest_path)]})
