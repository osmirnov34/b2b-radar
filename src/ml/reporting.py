"""Read-only, checksum-bound loading and reporting for completed ML artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from src.ml.cleaning_dataset import DatasetCleaningManifest
from src.ml.clustering import ClusteringManifest, ClusterSummary
from src.ml.corpus import CorpusManifest
from src.ml.inspection import DatasetInspection
from src.ml.outlier_reassignment import FinalClusterSummary, OutlierReassignmentManifest
from src.ml.semantic_deduplication import SemanticDeduplicationManifest
from src.ml.splitting import DatasetSplitManifest, SplitName
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


class DataSourceReference(_ReportModel):
    """Identify the persisted manifest field from which a metric originates."""

    path: str
    field: str


class ReassignmentStatus(StrEnum):
    NOT_RUN = "not_run"
    ELIGIBLE = "eligible"
    EXCLUDED = "excluded"


class ClusterCard(_ReportModel):
    """Summarize one topic without mixing incompatible assignment scores."""

    topic_id: int = Field(ge=0)
    name: str
    keywords: list[str]
    original_records: int = Field(ge=1)
    final_records: int = Field(ge=1)
    final_corpus_share: float = Field(gt=0, le=1)
    comments: int = Field(ge=0)
    replies: int = Field(ge=0)
    languages: dict[str, int]
    unique_videos: int = Field(ge=0)
    original_mean_probability: float = Field(ge=0, le=1)
    original_median_probability: float = Field(ge=0, le=1)
    original_minimum_probability: float = Field(ge=0, le=1)
    representative_records: int = Field(ge=0)
    reassigned_outliers: int = Field(ge=0)
    mean_reassignment_similarity: float | None = Field(default=None, ge=-1, le=1)
    minimum_reassignment_similarity: float | None = Field(default=None, ge=-1, le=1)
    reassignment_status: ReassignmentStatus
    reassignment_exclusion_reason: str | None = None
    pipeline_status: str
    warnings: list[str]
    sources: dict[str, DataSourceReference]


class ProcessingFlowStep(_ReportModel):
    stage: str
    records: int = Field(ge=0)
    note: str = ""


class DataScope(StrEnum):
    ALL = "all"
    DEVELOPMENT = "development"
    VALIDATION = "validation"
    TEST = "test"


class DataEntity(StrEnum):
    PARENT_COMMENT = "parent_comment"
    REPLY = "reply"
    TEXT_UNIT = "text_unit"
    CLUSTER_ASSIGNMENT = "cluster_assignment"


class DataLineageMetric(_ReportModel):
    """Describe one aggregate count together with its scope, entity, and source."""

    key: str
    label: str
    records: int = Field(ge=0)
    scope: DataScope
    entity: DataEntity
    source: DataSourceReference
    description: str


class DataLineageCheck(_ReportModel):
    """Record one successfully validated relationship between pipeline stages."""

    name: str
    expression: str
    passed: bool
    details: str


class DataLineage(_ReportModel):
    """Contain scope-aware flow metrics and their validated provenance checks."""

    metrics: list[DataLineageMetric]
    checks: list[DataLineageCheck]
    cleaning_removed_by_reason: dict[str, int]
    semantic_duplicates_removed: int = Field(ge=0)

    def metric(self, key: str) -> DataLineageMetric:
        """Return a metric by its stable key.

        Args:
            key: Machine-readable metric identifier from ``build_data_lineage``.

        Returns:
            The matching scope-aware metric.

        Raises:
            KeyError: If the lineage does not contain the requested metric.

        """
        try:
            return next(metric for metric in self.metrics if metric.key == key)
        except StopIteration as exc:
            msg = f"unknown data-lineage metric: {key}"
            raise KeyError(msg) from exc


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


@dataclass(frozen=True)
class _ResolvedClusterCounts:
    final_records: int
    comments: int
    replies: int
    languages: dict[str, int]
    unique_videos: int
    reassigned_outliers: int = 0
    mean_reassignment_similarity: float | None = None
    minimum_reassignment_similarity: float | None = None


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


def _load_cluster_summaries(artifacts: AnalysisArtifacts) -> tuple[ClusterSummary, ...]:
    """Load checksum-verified original HDBSCAN summaries for reporting."""
    path = _verified_path(
        artifacts.run_dir,
        artifacts.clustering.summary_path,
        artifacts.clustering.summary_sha256,
    )
    with path.open(encoding="utf-8") as source:
        return tuple(ClusterSummary.model_validate_json(line) for line in source if line.strip())


def _load_final_cluster_summaries(artifacts: AnalysisArtifacts) -> tuple[FinalClusterSummary, ...]:
    """Load checksum-verified post-reassignment summaries when that stage exists."""
    if artifacts.reassignment is None:
        return ()
    path = _verified_path(
        artifacts.run_dir,
        artifacts.reassignment.summary_path,
        artifacts.reassignment.summary_sha256,
    )
    with path.open(encoding="utf-8") as source:
        return tuple(FinalClusterSummary.model_validate_json(line) for line in source if line.strip())


def _validate_cluster_summary_ids(
    topics: tuple[TopicRepresentation, ...],
    originals: tuple[ClusterSummary, ...],
    finals: tuple[FinalClusterSummary, ...],
) -> None:
    """Require summary rows to follow normalized contiguous topic IDs."""
    expected_ids = list(range(len(topics)))
    if [summary.cluster_id for summary in originals] != expected_ids:
        msg = "original cluster summaries do not match normalized topic IDs"
        raise ValueError(msg)
    if finals and [summary.topic_id for summary in finals] != expected_ids:
        msg = "final cluster summaries do not match normalized topic IDs"
        raise ValueError(msg)


def _resolve_cluster_counts(
    topic_id: int,
    original: ClusterSummary,
    final: FinalClusterSummary | None,
) -> _ResolvedClusterCounts:
    """Resolve final counts while preserving the original HDBSCAN statistics."""
    if final is None:
        result = _ResolvedClusterCounts(
            final_records=original.records,
            comments=original.comments,
            replies=original.replies,
            languages=original.languages,
            unique_videos=original.unique_videos,
        )
    else:
        if final.original_records != original.records:
            msg = f"topic {topic_id} original count changed during reassignment"
            raise ValueError(msg)
        result = _ResolvedClusterCounts(
            final_records=final.final_records,
            comments=final.comments,
            replies=final.replies,
            languages=final.languages,
            unique_videos=final.unique_videos,
            reassigned_outliers=final.reassigned_outliers,
            mean_reassignment_similarity=final.mean_reassignment_similarity,
            minimum_reassignment_similarity=final.minimum_reassignment_similarity,
        )
    if result.comments + result.replies != result.final_records:
        msg = f"topic {topic_id} comment and reply counts do not match final records"
        raise ValueError(msg)
    return result


def _reassignment_details(
    artifacts: AnalysisArtifacts,
    topic_id: int,
) -> tuple[ReassignmentStatus, str | None]:
    """Describe whether one topic was eligible for outlier reassignment."""
    if artifacts.reassignment is None:
        return ReassignmentStatus.NOT_RUN, None
    if topic_id in artifacts.reassignment.eligible_topics:
        return ReassignmentStatus.ELIGIBLE, None
    reason = artifacts.reassignment.ineligible_topics.get(topic_id, "reason not recorded")
    return ReassignmentStatus.EXCLUDED, reason


def _cluster_card_sources(artifacts: AnalysisArtifacts, topic_id: int) -> dict[str, DataSourceReference]:
    """Return manifest rows that support a cluster card."""
    sources = {
        "clustering": DataSourceReference(
            path=artifacts.clustering.summary_path,
            field=f"cluster_id={topic_id}",
        ),
        "topic": DataSourceReference(
            path=artifacts.topics_manifest.representations_path,
            field=f"topic_id={topic_id}",
        ),
    }
    if artifacts.reassignment is not None:
        sources["reassignment"] = DataSourceReference(
            path=artifacts.reassignment.summary_path,
            field=f"topic_id={topic_id}",
        )
    return sources


def _build_cluster_card(
    artifacts: AnalysisArtifacts,
    topic: TopicRepresentation,
    original: ClusterSummary,
    final: FinalClusterSummary | None,
) -> ClusterCard:
    """Build one validated aggregate card from aligned stage summaries."""
    if topic.records != original.records:
        msg = f"topic {topic.topic_id} representation count does not match clustering summary"
        raise ValueError(msg)
    counts = _resolve_cluster_counts(topic.topic_id, original, final)
    reassignment_status, exclusion_reason = _reassignment_details(artifacts, topic.topic_id)
    warnings = []
    if original.mean_probability < artifacts.clustering.config.minimum_probability:
        warnings.append("original mean membership probability is below the configured quality threshold")
    if reassignment_status == ReassignmentStatus.EXCLUDED:
        warnings.append("topic was excluded from outlier reassignment")
    return ClusterCard(
        topic_id=topic.topic_id,
        name=topic.name,
        keywords=[keyword.term for keyword in topic.keywords],
        original_records=original.records,
        final_records=counts.final_records,
        final_corpus_share=counts.final_records / artifacts.summary.records,
        comments=counts.comments,
        replies=counts.replies,
        languages=dict(sorted(counts.languages.items(), key=lambda item: (-item[1], item[0]))),
        unique_videos=counts.unique_videos,
        original_mean_probability=original.mean_probability,
        original_median_probability=original.median_probability,
        original_minimum_probability=original.minimum_probability,
        representative_records=len(topic.representative_indices),
        reassigned_outliers=counts.reassigned_outliers,
        mean_reassignment_similarity=counts.mean_reassignment_similarity,
        minimum_reassignment_similarity=counts.minimum_reassignment_similarity,
        reassignment_status=reassignment_status,
        reassignment_exclusion_reason=exclusion_reason,
        pipeline_status=artifacts.summary.pipeline_status,
        warnings=warnings,
        sources=_cluster_card_sources(artifacts, topic.topic_id),
    )


def build_cluster_cards(artifacts: AnalysisArtifacts) -> list[ClusterCard]:
    """Build a checksum-bound, aggregate card for every topic.

    Original HDBSCAN membership probabilities and post-reassignment cosine
    similarities remain in separate fields because their scales have different
    meanings. Raw comments and author identifiers are never loaded.

    Args:
        artifacts: Checksum-verified aggregate artifacts for one pipeline run.

    Returns:
        Cards ordered by contiguous normalized ``topic_id``.

    Raises:
        FileNotFoundError: If a required cluster summary is missing.
        ValueError: If summary checksums, topic IDs, or record counts disagree.

    """
    original_summaries = _load_cluster_summaries(artifacts)
    final_summaries = _load_final_cluster_summaries(artifacts)
    _validate_cluster_summary_ids(artifacts.topics, original_summaries, final_summaries)
    original_by_id = {summary.cluster_id: summary for summary in original_summaries}
    final_by_id = {summary.topic_id: summary for summary in final_summaries}
    cards = [
        _build_cluster_card(
            artifacts,
            topic,
            original_by_id[topic.topic_id],
            final_by_id.get(topic.topic_id),
        )
        for topic in artifacts.topics
    ]
    if sum(card.final_records for card in cards) + artifacts.summary.outliers != artifacts.summary.records:
        msg = "cluster cards and outliers do not account for the final corpus"
        raise ValueError(msg)
    return cards


def get_cluster_card(cards: list[ClusterCard], topic_id: int) -> ClusterCard:
    """Return one cluster card by topic ID.

    Args:
        cards: Cards produced by ``build_cluster_cards``.
        topic_id: Non-negative normalized topic identifier.

    Returns:
        The matching cluster card.

    Raises:
        KeyError: If no card exists for ``topic_id``.

    """
    try:
        return next(card for card in cards if card.topic_id == topic_id)
    except StopIteration as exc:
        msg = f"unknown topic ID: {topic_id}"
        raise KeyError(msg) from exc


def _lineage_check(name: str, expression: str, passed: bool, details: str) -> DataLineageCheck:
    """Create a passed check and fail immediately when its invariant is false."""
    if not passed:
        msg = f"data-lineage invariant failed: {name}: {details}"
        raise ValueError(msg)
    return DataLineageCheck(name=name, expression=expression, passed=True, details=details)


def build_data_lineage(artifacts: AnalysisArtifacts) -> DataLineage:
    """Build and validate an auditable, scope-aware record flow for one run.

    Args:
        artifacts: Checksum-verified aggregate artifacts loaded for one pipeline run.

    Returns:
        Metrics with explicit dataset scopes and source fields, removal counts, and
        the successfully evaluated cross-stage invariants.

    Raises:
        FileNotFoundError: If a required stage manifest or referenced artifact is missing.
        ValueError: If checksums, row counts, flattening, or stage relationships disagree.

    """
    import numpy as np

    inspection_path = artifacts.run_dir / "01-inspection/dataset-profile.json"
    split_path = artifacts.run_dir / "02-split/split-manifest.json"
    cleaning_path = artifacts.run_dir / "04-cleaning/cleaning-manifest.json"
    deduplication_path = artifacts.run_dir / "06-deduplication/semantic-deduplication-manifest.json"
    inspection = DatasetInspection.model_validate_json(inspection_path.read_text(encoding="utf-8"))
    split = DatasetSplitManifest.model_validate_json(split_path.read_text(encoding="utf-8"))
    cleaning = DatasetCleaningManifest.model_validate_json(cleaning_path.read_text(encoding="utf-8"))
    deduplication = SemanticDeduplicationManifest.model_validate_json(
        deduplication_path.read_text(encoding="utf-8"),
    )

    _verified_path(artifacts.run_dir, cleaning.source_path, split.output_sha256[SplitName.DEVELOPMENT])
    _verified_path(artifacts.run_dir, cleaning.output_path, cleaning.output_sha256)
    _verified_path(artifacts.run_dir, deduplication.records_path, cleaning.output_sha256)
    checks = [
        _lineage_check(
            "inspection_to_split",
            "inspection.contract_valid == split.stats.input_records",
            inspection.contract_valid == split.stats.input_records,
            f"{inspection.contract_valid} == {split.stats.input_records}",
        ),
        _lineage_check(
            "inspection_checksum",
            "inspection.sha256 == split.source_sha256",
            inspection.sha256 == split.source_sha256,
            f"inspection={inspection.sha256[:12]}..., split={split.source_sha256[:12]}...",
        ),
        _lineage_check(
            "development_to_cleaning",
            "split development written == cleaning input rows == cleaning input comments",
            split.stats.written_records[SplitName.DEVELOPMENT]
            == cleaning.stats.input_rows
            == cleaning.stats.input_comments,
            (
                f"{split.stats.written_records[SplitName.DEVELOPMENT]} == "
                f"{cleaning.stats.input_rows} == {cleaning.stats.input_comments}"
            ),
        ),
        _lineage_check(
            "flattening",
            "cleaning input comments + replies == input text units",
            cleaning.stats.input_comments + cleaning.stats.input_replies == cleaning.stats.input_text_units,
            (
                f"{cleaning.stats.input_comments} + {cleaning.stats.input_replies} "
                f"== {cleaning.stats.input_text_units}"
            ),
        ),
        _lineage_check(
            "cleaning_to_deduplication",
            "cleaning output text units == semantic deduplication input",
            cleaning.stats.output_text_units == deduplication.result.n_input,
            f"{cleaning.stats.output_text_units} == {deduplication.result.n_input}",
        ),
        _lineage_check(
            "deduplication_to_corpus",
            "semantic deduplication kept == final corpus records",
            deduplication.result.n_kept == artifacts.corpus.stats.output_records,
            f"{deduplication.result.n_kept} == {artifacts.corpus.stats.output_records}",
        ),
        _lineage_check(
            "cleaning_manifest_checksum",
            "corpus cleaning manifest checksum == actual cleaning manifest checksum",
            artifacts.corpus.cleaning_manifest_sha256 == _sha256_file(cleaning_path),
            (
                f"corpus={artifacts.corpus.cleaning_manifest_sha256[:12]}..., "
                f"actual={_sha256_file(cleaning_path)[:12]}..."
            ),
        ),
        _lineage_check(
            "deduplication_manifest_checksum",
            "corpus deduplication manifest checksum == actual deduplication manifest checksum",
            artifacts.corpus.deduplication_manifest_sha256 == _sha256_file(deduplication_path),
            (
                f"corpus={artifacts.corpus.deduplication_manifest_sha256[:12]}..., "
                f"actual={_sha256_file(deduplication_path)[:12]}..."
            ),
        ),
        _lineage_check(
            "corpus_to_assignments",
            "assigned records + outliers == final corpus records",
            len(artifacts.labels) == artifacts.corpus.stats.output_records,
            f"{len(artifacts.labels)} == {artifacts.corpus.stats.output_records}",
        ),
    ]
    final_outliers = int(np.count_nonzero(artifacts.labels == -1))
    final_assigned = len(artifacts.labels) - final_outliers
    if artifacts.reassignment is not None:
        reassignment_metrics = artifacts.reassignment.metrics
        checks.extend(
            [
                _lineage_check(
                    "reassignment_input",
                    "original outliers == reassigned + remaining outliers",
                    reassignment_metrics.original_outliers
                    == reassignment_metrics.reassigned_outliers + reassignment_metrics.remaining_outliers,
                    (
                        f"{reassignment_metrics.original_outliers} == "
                        f"{reassignment_metrics.reassigned_outliers} + {reassignment_metrics.remaining_outliers}"
                    ),
                ),
                _lineage_check(
                    "reassignment_output",
                    "manifest remaining outliers == final label outliers",
                    reassignment_metrics.remaining_outliers == final_outliers,
                    f"{reassignment_metrics.remaining_outliers} == {final_outliers}",
                ),
            ],
        )

    lineage_metrics = [
        DataLineageMetric(
            key="all_valid_parents",
            label="Valid parent comments",
            records=inspection.contract_valid,
            scope=DataScope.ALL,
            entity=DataEntity.PARENT_COMMENT,
            source=DataSourceReference(path=str(inspection_path), field="contract_valid"),
            description="Contract-valid top-level comments in the complete input dataset.",
        ),
    ]
    lineage_metrics.extend(
        (
            DataLineageMetric(
                key=f"{split_name.value}_parents",
                label=f"{split_name.value.title()} parent comments",
                records=split.stats.written_records[split_name],
                scope=DataScope(split_name.value),
                entity=DataEntity.PARENT_COMMENT,
                source=DataSourceReference(
                    path=str(split_path),
                    field=f"stats.written_records.{split_name.value}",
                ),
                description="Top-level comments retained in this leakage-safe split.",
            )
            for split_name in SplitName
        ),
    )
    lineage_metrics.extend(
        [
            DataLineageMetric(
                key="development_replies",
                label="Development nested replies",
                records=cleaning.stats.input_replies,
                scope=DataScope.DEVELOPMENT,
                entity=DataEntity.REPLY,
                source=DataSourceReference(path=str(cleaning_path), field="stats.input_replies"),
                description="Nested replies attached to development parent comments.",
            ),
            DataLineageMetric(
                key="development_flattened",
                label="Flattened development text units",
                records=cleaning.stats.input_text_units,
                scope=DataScope.DEVELOPMENT,
                entity=DataEntity.TEXT_UNIT,
                source=DataSourceReference(path=str(cleaning_path), field="stats.input_text_units"),
                description="Development parent comments and replies before cleaning.",
            ),
            DataLineageMetric(
                key="development_cleaned",
                label="Cleaned development text units",
                records=cleaning.stats.output_text_units,
                scope=DataScope.DEVELOPMENT,
                entity=DataEntity.TEXT_UNIT,
                source=DataSourceReference(path=str(cleaning_path), field="stats.output_text_units"),
                description="Development text units retained after deterministic cleaning and exact deduplication.",
            ),
            DataLineageMetric(
                key="development_semantic_deduplicated",
                label="Semantically deduplicated text units",
                records=deduplication.result.n_kept,
                scope=DataScope.DEVELOPMENT,
                entity=DataEntity.TEXT_UNIT,
                source=DataSourceReference(path=str(deduplication_path), field="result.n_kept"),
                description="Development text units retained after semantic near-duplicate removal.",
            ),
            DataLineageMetric(
                key="development_final_corpus",
                label="Final development corpus",
                records=artifacts.corpus.stats.output_records,
                scope=DataScope.DEVELOPMENT,
                entity=DataEntity.TEXT_UNIT,
                source=DataSourceReference(
                    path=str(artifacts.run_dir / "07-corpus/corpus-manifest.json"),
                    field="stats.output_records",
                ),
                description="Row-aligned corpus consumed by reduction and clustering.",
            ),
            DataLineageMetric(
                key="development_assigned",
                label="Assigned to topics",
                records=final_assigned,
                scope=DataScope.DEVELOPMENT,
                entity=DataEntity.CLUSTER_ASSIGNMENT,
                source=DataSourceReference(
                    path=str(
                        artifacts.run_dir / "11-reassignment/outlier-reassignment-manifest.json"
                        if artifacts.reassignment is not None
                        else artifacts.run_dir / "09-clustering/clustering-manifest.json"
                    ),
                    field="final_labels != -1",
                ),
                description="Final corpus rows assigned to a topic after optional reassignment.",
            ),
            DataLineageMetric(
                key="development_outliers",
                label="Remaining outliers",
                records=final_outliers,
                scope=DataScope.DEVELOPMENT,
                entity=DataEntity.CLUSTER_ASSIGNMENT,
                source=DataSourceReference(
                    path=str(
                        artifacts.run_dir / "11-reassignment/outlier-reassignment-manifest.json"
                        if artifacts.reassignment is not None
                        else artifacts.run_dir / "09-clustering/clustering-manifest.json"
                    ),
                    field="final_labels == -1",
                ),
                description="Final corpus rows that remain outside every topic.",
            ),
        ],
    )
    return DataLineage(
        metrics=lineage_metrics,
        checks=checks,
        cleaning_removed_by_reason={reason.value: count for reason, count in cleaning.stats.removed_by_reason.items()},
        semantic_duplicates_removed=deduplication.result.n_removed,
    )


def processing_flow(artifacts: AnalysisArtifacts) -> list[ProcessingFlowStep]:
    """Return a linear development-only view retained for API compatibility.

    Args:
        artifacts: Checksum-verified aggregate artifacts for one pipeline run.

    Returns:
        Development parent, flattened, cleaned, deduplicated, and corpus counts.

    Raises:
        FileNotFoundError: If lineage inputs are incomplete.
        ValueError: If ``build_data_lineage`` detects inconsistent artifacts.

    """
    lineage = build_data_lineage(artifacts)
    return [
        ProcessingFlowStep(
            stage=metric.label,
            records=metric.records,
            note=f"scope={metric.scope.value}; source={Path(metric.source.path).name}:{metric.source.field}",
        )
        for metric in lineage.metrics
        if metric.key
        in {
            "development_parents",
            "development_flattened",
            "development_cleaned",
            "development_semantic_deduplicated",
            "development_final_corpus",
        }
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
