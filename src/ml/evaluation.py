"""Multi-signal evaluation for fixed topic clustering artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field

from src.ml.clustering import ClusteringManifest, HDBSCANConfig
from src.ml.corpus import CorpusManifest, CorpusRecord
from src.ml.dimensionality_reduction import ReductionArtifactManifest, ReductionMode, UMAPConfig
from src.ml.outlier_reassignment import OutlierReassignmentManifest
from src.ml.topic_representation import RepresentativeIndices, TopicRepresentation, TopicRepresentationManifest

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

_EMAIL_PATTERN = re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b")
_PHONE_PATTERN = re.compile(r"(?<!\d)(?:\+?\d[\s().-]?){7,}\d(?!\d)")
_MINIMUM_GEOMETRY_CLUSTERS = 2


class _EvaluationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EvaluationStatus(StrEnum):
    PASS = "pass"
    PASS_WITH_WARNINGS = "pass_with_warnings"
    FAIL = "fail"


class EvaluationConfig(_EvaluationModel):
    schema_version: int = 1
    geometry_sample_size: int = Field(default=10_000, ge=2)
    bootstrap_runs: int = Field(default=0, ge=0, le=100)
    bootstrap_share: float = Field(default=0.8, gt=0, le=1)
    manual_topics: int = Field(default=20, ge=1)
    manual_examples_per_topic: int = Field(default=5, ge=1)
    manual_outliers: int = Field(default=20, ge=0)
    minimum_manual_annotations: int = Field(default=100, ge=1)
    random_seed: int = 42
    maximum_final_outlier_share: float = Field(default=0.5, ge=0, le=1)
    minimum_stability_ari: float = Field(default=0.6, ge=-1, le=1)
    minimum_manual_precision: float = Field(default=0.75, ge=0, le=1)
    minimum_business_relevance: float = Field(default=0.6, ge=0, le=1)
    maximum_topic_keyword_jaccard: float = Field(default=0.7, ge=0, le=1)
    maximum_reassignment_degradation: float = Field(default=0.05, ge=0, le=2)
    validation_completed: bool = False


class GeometryMetrics(_EvaluationModel):
    evaluated_records: int = Field(ge=0)
    clusters: int = Field(ge=0)
    silhouette: float | None = Field(default=None, ge=-1, le=1)
    davies_bouldin: float | None = Field(default=None, ge=0)
    calinski_harabasz: float | None = Field(default=None, ge=0)
    mean_intra_cluster_similarity: float | None = Field(default=None, ge=-1, le=1)
    maximum_centroid_similarity: float | None = Field(default=None, ge=-1, le=1)


class BootstrapRun(_EvaluationModel):
    run: int = Field(ge=0)
    records: int = Field(ge=2)
    ari: float = Field(ge=-1, le=1)
    nmi: float = Field(ge=0, le=1)


class ClusterMatch(_EvaluationModel):
    run: int = Field(ge=0)
    reference_cluster: int = Field(ge=0)
    candidate_cluster: int = Field(ge=0)
    overlap: int = Field(ge=1)
    reference_retention: float = Field(gt=0, le=1)
    split_detected: bool
    merge_detected: bool


class TopicEvaluation(_EvaluationModel):
    topic_id: int = Field(ge=0)
    records: int = Field(ge=1)
    keyword_count: int = Field(ge=0)
    representative_count: int = Field(ge=0)
    suspicious_term_count: int = Field(ge=0)
    mean_representative_similarity: float | None = Field(default=None, ge=-1, le=1)


class ManualReviewRecord(_EvaluationModel):
    record_index: int = Field(ge=0)
    topic_id: int = Field(ge=-1)
    sample_kind: str
    text: str
    confidence: float = Field(ge=0, le=1)


class ManualAnnotation(_EvaluationModel):
    record_index: int = Field(ge=0)
    topic_matches: bool
    topic_clear: bool
    business_relevant: bool
    reassignment_correct: bool | None = None
    contains_sensitive_data: bool = False
    merge_candidate: bool = False
    split_candidate: bool = False
    reviewer: str = ""
    note: str = ""


class ManualMetrics(_EvaluationModel):
    annotations: int = Field(ge=0)
    topic_precision: float | None = Field(default=None, ge=0, le=1)
    clear_topic_share: float | None = Field(default=None, ge=0, le=1)
    business_relevance_share: float | None = Field(default=None, ge=0, le=1)
    reassignment_precision: float | None = Field(default=None, ge=0, le=1)
    sensitive_data_flags: int = Field(ge=0)
    merge_candidates: int = Field(ge=0)
    split_candidates: int = Field(ge=0)


class EvaluationMetrics(_EvaluationModel):
    records: int = Field(ge=0)
    topics: int = Field(ge=0)
    original_outlier_share: float = Field(ge=0, le=1)
    final_outlier_share: float = Field(ge=0, le=1)
    changed_label_share: float = Field(ge=0, le=1)
    original_geometry: GeometryMetrics
    final_geometry: GeometryMetrics
    bootstrap_runs: list[BootstrapRun]
    mean_bootstrap_ari: float | None = Field(default=None, ge=-1, le=1)
    mean_bootstrap_nmi: float | None = Field(default=None, ge=0, le=1)
    manual: ManualMetrics
    suspicious_topic_terms: int = Field(ge=0)
    status: EvaluationStatus
    preliminary: bool


class EvaluationManifest(_EvaluationModel):
    schema_version: int = 1
    corpus_manifest_path: str
    corpus_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    clustering_manifest_path: str
    clustering_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    topic_manifest_path: str
    topic_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reassignment_manifest_path: str
    reassignment_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    config: EvaluationConfig
    metrics_path: str
    metrics_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    topic_evaluation_path: str
    topic_evaluation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cluster_matching_path: str
    cluster_matching_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    manual_review_sample_path: str
    manual_review_sample_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    manual_review_template_path: str
    manual_review_template_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    manual_annotations_path: str | None = None
    manual_annotations_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    warnings: list[str]
    created_at: datetime


class GeometryBackend(Protocol):
    def evaluate(self, vectors: NDArray[np.float32], labels: NDArray[np.int64]) -> GeometryMetrics: ...


class StabilityBackend(Protocol):
    def evaluate(
        self,
        vectors: NDArray[np.float32],
        labels: NDArray[np.int64],
        config: EvaluationConfig,
        umap_config: UMAPConfig,
        hdbscan_config: HDBSCANConfig,
    ) -> tuple[list[BootstrapRun], list[ClusterMatch]]: ...


class SklearnGeometryBackend:
    def evaluate(self, vectors: NDArray[np.float32], labels: NDArray[np.int64]) -> GeometryMetrics:
        import numpy as np

        clustered = labels >= 0
        selected_vectors = vectors[clustered]
        selected_labels = labels[clustered]
        clusters = sorted({int(label) for label in selected_labels})
        if len(clusters) < _MINIMUM_GEOMETRY_CLUSTERS or len(selected_vectors) <= len(clusters):
            return GeometryMetrics(evaluated_records=len(selected_vectors), clusters=len(clusters))
        from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

        centroids = []
        intra: list[float] = []
        normalized = selected_vectors / np.linalg.norm(selected_vectors, axis=1, keepdims=True)
        for cluster_id in clusters:
            group = normalized[selected_labels == cluster_id]
            centroid = np.mean(group, axis=0)
            centroid /= np.linalg.norm(centroid)
            centroids.append(centroid)
            intra.extend(float(value) for value in group @ centroid)
        centroid_matrix = np.asarray(centroids, dtype=np.float32)
        similarities = centroid_matrix @ centroid_matrix.T
        np.fill_diagonal(similarities, -1)
        return GeometryMetrics(
            evaluated_records=len(selected_vectors),
            clusters=len(clusters),
            silhouette=float(silhouette_score(selected_vectors, selected_labels, metric="cosine")),
            davies_bouldin=float(davies_bouldin_score(selected_vectors, selected_labels)),
            calinski_harabasz=float(calinski_harabasz_score(selected_vectors, selected_labels)),
            mean_intra_cluster_similarity=float(np.mean(intra)),
            maximum_centroid_similarity=float(np.max(similarities)),
        )


class UMAPHDBSCANStabilityBackend:
    """Bootstrap backend; imports heavy optional dependencies only when runs are requested."""

    def evaluate(
        self,
        vectors: NDArray[np.float32],
        labels: NDArray[np.int64],
        config: EvaluationConfig,
        umap_config: UMAPConfig,
        hdbscan_config: HDBSCANConfig,
    ) -> tuple[list[BootstrapRun], list[ClusterMatch]]:
        if config.bootstrap_runs == 0:
            return [], []
        try:
            import hdbscan
            import numpy as np
            import umap
            from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        except ImportError as exc:  # pragma: no cover - optional analysis environment
            msg = "bootstrap evaluation requires umap-learn, hdbscan, and scikit-learn"
            raise RuntimeError(msg) from exc
        generator = np.random.default_rng(config.random_seed)
        runs = []
        matches = []
        sample_size = max(2, int(len(vectors) * config.bootstrap_share))
        neighbors = min(umap_config.n_neighbors, sample_size - 1)
        minimum_cluster_size = min(hdbscan_config.min_cluster_size, sample_size)
        for run in range(config.bootstrap_runs):
            indices = np.sort(generator.choice(len(vectors), size=sample_size, replace=False))
            reduced = umap.UMAP(
                n_neighbors=neighbors,
                n_components=umap_config.clustering_components,
                min_dist=umap_config.clustering_min_dist,
                metric=umap_config.metric,
                random_state=config.random_seed + run,
                low_memory=umap_config.low_memory,
                n_jobs=1,
                transform_seed=config.random_seed + run,
            ).fit_transform(vectors[indices])
            candidate = hdbscan.HDBSCAN(
                min_cluster_size=minimum_cluster_size,
                min_samples=hdbscan_config.min_samples,
                metric=hdbscan_config.metric,
                cluster_selection_method=hdbscan_config.cluster_selection_method,
                cluster_selection_epsilon=hdbscan_config.cluster_selection_epsilon,
                alpha=hdbscan_config.alpha,
                allow_single_cluster=hdbscan_config.allow_single_cluster,
                core_dist_n_jobs=hdbscan_config.threads,
            ).fit_predict(reduced)
            reference = labels[indices]
            runs.append(
                BootstrapRun(
                    run=run,
                    records=sample_size,
                    ari=float(adjusted_rand_score(reference, candidate)),
                    nmi=float(normalized_mutual_info_score(reference, candidate)),
                ),
            )
            matches.extend(match_clusters(reference, candidate, run=run))
        return runs, matches


GeometryFactory = Callable[[], GeometryBackend]
StabilityFactory = Callable[[], StabilityBackend]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_evaluation_indices(records: int, sample_size: int, seed: int) -> list[int]:
    import numpy as np

    actual = min(records, sample_size)
    if actual == records:
        return list(range(records))
    generator = np.random.default_rng(seed)
    return sorted(int(index) for index in generator.choice(records, size=actual, replace=False))


def match_clusters(
    reference: NDArray[np.integer],
    candidate: NDArray[np.integer],
    *,
    run: int,
) -> list[ClusterMatch]:
    import numpy as np

    if reference.shape != candidate.shape:
        msg = "reference and candidate labels must have identical shapes"
        raise ValueError(msg)
    reference_ids = sorted({int(label) for label in reference if label >= 0})
    candidate_ids = sorted({int(label) for label in candidate if label >= 0})
    if not reference_ids or not candidate_ids:
        return []
    matrix = np.zeros((len(reference_ids), len(candidate_ids)), dtype=np.int64)
    for left, reference_id in enumerate(reference_ids):
        for right, candidate_id in enumerate(candidate_ids):
            matrix[left, right] = np.count_nonzero((reference == reference_id) & (candidate == candidate_id))
    from scipy.optimize import linear_sum_assignment

    rows, columns = linear_sum_assignment(-matrix)
    matches = []
    for row, column in zip(rows, columns, strict=True):
        overlap = int(matrix[row, column])
        if overlap == 0:
            continue
        size = int(np.count_nonzero(reference == reference_ids[row]))
        reference_fragments = int(np.count_nonzero(matrix[row]))
        candidate_sources = int(np.count_nonzero(matrix[:, column]))
        matches.append(
            ClusterMatch(
                run=run,
                reference_cluster=reference_ids[row],
                candidate_cluster=candidate_ids[column],
                overlap=overlap,
                reference_retention=overlap / size,
                split_detected=reference_fragments > 1,
                merge_detected=candidate_sources > 1,
            ),
        )
    return matches


def _validate_contracts(
    corpus_manifest_path: Path,
    clustering_manifest_path: Path,
    topic_manifest_path: Path,
    reassignment_manifest_path: Path,
    *,
    require_reduction: bool,
) -> tuple[
    CorpusManifest,
    ClusteringManifest,
    ReductionArtifactManifest | None,
    TopicRepresentationManifest,
    OutlierReassignmentManifest,
]:
    corpus = CorpusManifest.model_validate_json(corpus_manifest_path.read_text(encoding="utf-8"))
    clustering = ClusteringManifest.model_validate_json(clustering_manifest_path.read_text(encoding="utf-8"))
    topics = TopicRepresentationManifest.model_validate_json(topic_manifest_path.read_text(encoding="utf-8"))
    reassignment = OutlierReassignmentManifest.model_validate_json(
        reassignment_manifest_path.read_text(encoding="utf-8"),
    )
    reduction_path = Path(clustering.reduction_manifest_path)
    reduction = (
        ReductionArtifactManifest.model_validate_json(reduction_path.read_text(encoding="utf-8"))
        if require_reduction
        else None
    )
    checks = (
        (_sha256_file(corpus_manifest_path), clustering.corpus_manifest_sha256, "corpus manifest"),
        (_sha256_file(clustering_manifest_path), topics.clustering_manifest_sha256, "clustering manifest"),
        (_sha256_file(topic_manifest_path), reassignment.topic_manifest_sha256, "topic manifest"),
    )
    for actual, expected, name in checks:
        if actual != expected:
            msg = f"{name} checksum does not match its downstream manifest"
            raise ValueError(msg)
    if reduction is not None:
        if _sha256_file(reduction_path) != clustering.reduction_manifest_sha256:
            msg = "reduction manifest checksum does not match its downstream manifest"
            raise ValueError(msg)
        if reduction.mode != ReductionMode.CLUSTERING:
            msg = "bootstrap evaluation requires the clustering UMAP manifest"
            raise ValueError(msg)
    return corpus, clustering, reduction, topics, reassignment


def _load_vector(path: Path, expected_hash: str, records: int, *, integer: bool) -> NDArray[np.generic]:
    import numpy as np

    if _sha256_file(path) != expected_hash:
        msg = f"array checksum does not match manifest: {path}"
        raise ValueError(msg)
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if values.shape != (records,):
        msg = f"array shape {values.shape} does not match {(records,)}"
        raise ValueError(msg)
    if integer and values.dtype.kind not in "iu":
        msg = "labels array must use an integer dtype"
        raise ValueError(msg)
    if not integer and (not np.isfinite(values).all() or np.any((values < 0) | (values > 1))):
        msg = "confidence array must contain finite values in [0, 1]"
        raise ValueError(msg)
    return cast("NDArray[np.generic]", values)


def _representative_similarities(manifest: TopicRepresentationManifest) -> dict[int, float | None]:
    path = Path(manifest.representative_indices_path)
    if _sha256_file(path) != manifest.representative_indices_sha256:
        msg = "representative indices checksum does not match topic manifest"
        raise ValueError(msg)
    values = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                item = RepresentativeIndices.model_validate_json(line)
                values[item.topic_id] = (
                    sum(item.centroid_similarities) / len(item.centroid_similarities)
                    if item.centroid_similarities
                    else None
                )
    return values


def _topic_evaluations(path: Path, manifest: TopicRepresentationManifest) -> list[TopicEvaluation]:
    if _sha256_file(path) != manifest.representations_sha256:
        msg = "topic representations checksum does not match topic manifest"
        raise ValueError(msg)
    results = []
    similarities = _representative_similarities(manifest)
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            item = TopicRepresentation.model_validate_json(line)
            suspicious = sum(
                bool(_EMAIL_PATTERN.search(keyword.term) or _PHONE_PATTERN.search(keyword.term))
                for keyword in item.keywords
            )
            results.append(
                TopicEvaluation(
                    topic_id=item.topic_id,
                    records=item.records,
                    keyword_count=len(item.keywords),
                    representative_count=len(item.representative_indices),
                    suspicious_term_count=suspicious,
                    mean_representative_similarity=similarities.get(item.topic_id),
                ),
            )
    return results


def _manual_metrics(path: Path | None, records: int) -> tuple[ManualMetrics, str | None]:
    if path is None:
        return ManualMetrics(
            annotations=0,
            sensitive_data_flags=0,
            merge_candidates=0,
            split_candidates=0,
        ), None
    annotations = [
        ManualAnnotation.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    indices = [item.record_index for item in annotations]
    if len(indices) != len(set(indices)):
        msg = "manual annotations contain duplicate record_index values"
        raise ValueError(msg)
    if any(index >= records for index in indices):
        msg = "manual annotation record_index is outside the evaluated corpus"
        raise ValueError(msg)
    count = len(annotations)
    reassigned = [item for item in annotations if item.reassignment_correct is not None]
    metrics = ManualMetrics(
        annotations=count,
        topic_precision=sum(item.topic_matches for item in annotations) / count if count else None,
        clear_topic_share=sum(item.topic_clear for item in annotations) / count if count else None,
        business_relevance_share=sum(item.business_relevant for item in annotations) / count if count else None,
        reassignment_precision=(
            sum(bool(item.reassignment_correct) for item in reassigned) / len(reassigned) if reassigned else None
        ),
        sensitive_data_flags=sum(item.contains_sensitive_data for item in annotations),
        merge_candidates=sum(item.merge_candidate for item in annotations),
        split_candidates=sum(item.split_candidate for item in annotations),
    )
    return metrics, _sha256_file(path)


def _verdict(
    config: EvaluationConfig,
    final_outlier_share: float,
    final_geometry: GeometryMetrics,
    original_geometry: GeometryMetrics,
    bootstrap_runs: list[BootstrapRun],
    manual: ManualMetrics,
    keyword_jaccard: float,
) -> tuple[EvaluationStatus, bool, list[str]]:
    warnings = []
    failures = []
    if final_outlier_share > config.maximum_final_outlier_share:
        failures.append("final outlier share exceeds threshold")
    if keyword_jaccard > config.maximum_topic_keyword_jaccard:
        warnings.append("topic keyword overlap exceeds threshold")
    mean_ari = sum(run.ari for run in bootstrap_runs) / len(bootstrap_runs) if bootstrap_runs else None
    if mean_ari is not None and mean_ari < config.minimum_stability_ari:
        failures.append("bootstrap ARI is below threshold")
    if original_geometry.silhouette is not None and final_geometry.silhouette is not None:
        degradation = original_geometry.silhouette - final_geometry.silhouette
        if degradation > config.maximum_reassignment_degradation:
            failures.append("reassignment degraded silhouette beyond threshold")
    preliminary = manual.annotations == 0 or not config.validation_completed
    if manual.annotations == 0:
        warnings.append("manual review is missing")
    else:
        if manual.annotations < config.minimum_manual_annotations:
            failures.append("manual review has fewer annotations than required")
        if manual.topic_precision is not None and manual.topic_precision < config.minimum_manual_precision:
            failures.append("manual topic precision is below threshold")
        if (
            manual.business_relevance_share is not None
            and manual.business_relevance_share < config.minimum_business_relevance
        ):
            failures.append("manual business relevance is below threshold")
    if not config.validation_completed:
        warnings.append("validation evaluation is not completed")
    if failures:
        return EvaluationStatus.FAIL, preliminary, failures + warnings
    if warnings or preliminary:
        return EvaluationStatus.PASS_WITH_WARNINGS, preliminary, warnings
    return EvaluationStatus.PASS, False, []


def _write_manual_sample(
    corpus_path: Path,
    source_labels: NDArray[np.int64],
    labels: NDArray[np.int64],
    confidence: NDArray[np.float32],
    config: EvaluationConfig,
    target: Path,
) -> None:
    per_topic: dict[int, list[ManualReviewRecord]] = {}
    outlier_samples: list[ManualReviewRecord] = []
    with corpus_path.open(encoding="utf-8") as source:
        index = 0
        for line in source:
            if not line.strip():
                continue
            if index >= len(labels):
                break
            label = int(labels[index])
            kind = "reassigned" if int(source_labels[index]) == -1 and label >= 0 else "topic_member"
            candidates = per_topic.setdefault(label, []) if 0 <= label < config.manual_topics else None
            include = candidates is not None and len(candidates) < config.manual_examples_per_topic
            replace = (
                candidates is not None
                and kind == "reassigned"
                and not any(item.sample_kind == "reassigned" for item in candidates)
            )
            if include or replace or (label == -1 and len(outlier_samples) < config.manual_outliers):
                record = CorpusRecord.model_validate_json(line)
                sample = ManualReviewRecord(
                    record_index=index,
                    topic_id=label,
                    sample_kind=kind,
                    text=record.clean_text,
                    confidence=float(confidence[index]),
                )
                if label == -1:
                    outlier_samples.append(sample.model_copy(update={"sample_kind": "remaining_outlier"}))
                elif replace and candidates is not None:
                    candidates[-1] = sample
                elif candidates is not None:
                    candidates.append(sample)
            index += 1
    selected = [item for topic_id in sorted(per_topic) for item in per_topic[topic_id]] + outlier_samples
    with target.open("w", encoding="utf-8") as output:
        for item in sorted(selected, key=lambda sample: sample.record_index):
            output.write(f"{item.model_dump_json()}\n")


def evaluate_topics(
    embeddings_path: Path,
    corpus_path: Path,
    source_labels_path: Path,
    final_labels_path: Path,
    final_confidence_path: Path,
    corpus_manifest_path: Path,
    clustering_manifest_path: Path,
    topic_manifest_path: Path,
    reassignment_manifest_path: Path,
    output_dir: Path,
    *,
    config: EvaluationConfig | None = None,
    manual_annotations_path: Path | None = None,
    geometry_factory: GeometryFactory | None = None,
    stability_factory: StabilityFactory | None = None,
    force: bool = False,
) -> EvaluationManifest:
    """Evaluate cluster geometry, stability, topics, reassignment, and optional human review."""
    import numpy as np

    active_config = config or EvaluationConfig()
    corpus, clustering, reduction, topics, reassignment = _validate_contracts(
        corpus_manifest_path,
        clustering_manifest_path,
        topic_manifest_path,
        reassignment_manifest_path,
        require_reduction=active_config.bootstrap_runs > 0,
    )
    records = reassignment.metrics.records
    if _sha256_file(embeddings_path) != corpus.final_embeddings_sha256:
        msg = "embeddings checksum does not match corpus manifest"
        raise ValueError(msg)
    if _sha256_file(corpus_path) != corpus.corpus_sha256:
        msg = "corpus checksum does not match corpus manifest"
        raise ValueError(msg)
    source_labels = cast(
        "NDArray[np.int64]",
        _load_vector(source_labels_path, clustering.labels_sha256, records, integer=True),
    )
    final_labels = cast(
        "NDArray[np.int64]",
        _load_vector(final_labels_path, reassignment.final_labels_sha256, records, integer=True),
    )
    final_confidence = cast(
        "NDArray[np.float32]",
        _load_vector(
            final_confidence_path,
            reassignment.final_confidence_sha256,
            records,
            integer=False,
        ),
    )
    embeddings = np.load(embeddings_path, mmap_mode="r", allow_pickle=False)
    if embeddings.shape[0] < records or embeddings.shape[1] != corpus.dimensions:
        msg = "embeddings shape does not match evaluation inputs"
        raise ValueError(msg)
    if not np.isfinite(embeddings[:records]).all() or np.any(np.linalg.norm(embeddings[:records], axis=1) == 0):
        msg = "embeddings must contain only finite, non-zero vectors"
        raise ValueError(msg)
    indices = deterministic_evaluation_indices(records, active_config.geometry_sample_size, active_config.random_seed)
    vectors = np.asarray(embeddings[indices], dtype=np.float32)
    original_sample_labels = np.asarray(source_labels[indices], dtype=np.int64)
    final_sample_labels = np.asarray(final_labels[indices], dtype=np.int64)
    geometry = (geometry_factory or SklearnGeometryBackend)()
    original_geometry = geometry.evaluate(vectors, original_sample_labels)
    final_geometry = geometry.evaluate(vectors, final_sample_labels)
    stability = (stability_factory or UMAPHDBSCANStabilityBackend)()
    bootstrap_runs, cluster_matches = stability.evaluate(
        vectors,
        final_sample_labels,
        active_config,
        reduction.config if reduction is not None else UMAPConfig(),
        clustering.config,
    )
    representations_path = Path(topics.representations_path)
    topic_evaluations = _topic_evaluations(representations_path, topics)
    manual, manual_sha = _manual_metrics(manual_annotations_path, records)
    final_outliers = int(np.count_nonzero(final_labels == -1))
    changed = int(np.count_nonzero(source_labels != final_labels))
    status, preliminary, warnings = _verdict(
        active_config,
        final_outliers / records,
        final_geometry,
        original_geometry,
        bootstrap_runs,
        manual,
        topics.quality.maximum_keyword_jaccard,
    )
    mean_ari = sum(run.ari for run in bootstrap_runs) / len(bootstrap_runs) if bootstrap_runs else None
    mean_nmi = sum(run.nmi for run in bootstrap_runs) / len(bootstrap_runs) if bootstrap_runs else None
    metrics = EvaluationMetrics(
        records=records,
        topics=topics.topics,
        original_outlier_share=clustering.metrics.outlier_share,
        final_outlier_share=final_outliers / records,
        changed_label_share=changed / records,
        original_geometry=original_geometry,
        final_geometry=final_geometry,
        bootstrap_runs=bootstrap_runs,
        mean_bootstrap_ari=mean_ari,
        mean_bootstrap_nmi=mean_nmi,
        manual=manual,
        suspicious_topic_terms=sum(item.suspicious_term_count for item in topic_evaluations),
        status=status,
        preliminary=preliminary,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "evaluation-metrics.json"
    topic_evaluation_path = output_dir / "topic-evaluation.jsonl"
    matching_path = output_dir / "cluster-matching.jsonl"
    manual_sample_path = output_dir / "manual-review-sample.jsonl"
    manual_template_path = output_dir / "manual-review-template.json"
    manifest_path = output_dir / "evaluation-manifest.json"
    report_path = output_dir / "evaluation-report.md"
    finals = (
        metrics_path,
        topic_evaluation_path,
        matching_path,
        manual_sample_path,
        manual_template_path,
        manifest_path,
        report_path,
    )
    existing = next((path for path in finals if path.exists()), None)
    if existing is not None and not force:
        msg = f"refusing to overwrite existing evaluation artifact: {existing}"
        raise FileExistsError(msg)
    temporary = {path: path.with_name(f".{path.name}.tmp") for path in finals}
    for path in temporary.values():
        path.unlink(missing_ok=True)
    try:
        temporary[metrics_path].write_text(f"{metrics.model_dump_json(indent=2)}\n", encoding="utf-8")
        with temporary[topic_evaluation_path].open("w", encoding="utf-8") as target:
            for item in topic_evaluations:
                target.write(f"{item.model_dump_json()}\n")
        with temporary[matching_path].open("w", encoding="utf-8") as target:
            for match in cluster_matches:
                target.write(f"{match.model_dump_json()}\n")
        _write_manual_sample(
            corpus_path,
            source_labels,
            final_labels,
            final_confidence,
            active_config,
            temporary[manual_sample_path],
        )
        template = {
            "record_index": 0,
            "topic_matches": None,
            "topic_clear": None,
            "business_relevant": None,
            "reassignment_correct": None,
            "contains_sensitive_data": False,
            "merge_candidate": False,
            "split_candidate": False,
            "reviewer": "",
            "note": "",
        }
        temporary[manual_template_path].write_text(f"{json.dumps(template, indent=2)}\n", encoding="utf-8")
        manifest = EvaluationManifest(
            corpus_manifest_path=str(corpus_manifest_path),
            corpus_manifest_sha256=_sha256_file(corpus_manifest_path),
            clustering_manifest_path=str(clustering_manifest_path),
            clustering_manifest_sha256=_sha256_file(clustering_manifest_path),
            topic_manifest_path=str(topic_manifest_path),
            topic_manifest_sha256=_sha256_file(topic_manifest_path),
            reassignment_manifest_path=str(reassignment_manifest_path),
            reassignment_manifest_sha256=_sha256_file(reassignment_manifest_path),
            config=active_config,
            metrics_path=str(metrics_path),
            metrics_sha256=_sha256_file(temporary[metrics_path]),
            topic_evaluation_path=str(topic_evaluation_path),
            topic_evaluation_sha256=_sha256_file(temporary[topic_evaluation_path]),
            cluster_matching_path=str(matching_path),
            cluster_matching_sha256=_sha256_file(temporary[matching_path]),
            manual_review_sample_path=str(manual_sample_path),
            manual_review_sample_sha256=_sha256_file(temporary[manual_sample_path]),
            manual_review_template_path=str(manual_template_path),
            manual_review_template_sha256=_sha256_file(temporary[manual_template_path]),
            manual_annotations_path=str(manual_annotations_path) if manual_annotations_path else None,
            manual_annotations_sha256=manual_sha,
            warnings=warnings,
            created_at=datetime.now(UTC),
        )
        temporary[manifest_path].write_text(f"{manifest.model_dump_json(indent=2)}\n", encoding="utf-8")
        temporary[report_path].write_text(_evaluation_report(metrics, warnings), encoding="utf-8")
        for final in (*finals[:-2], report_path, manifest_path):
            temporary[final].replace(final)
    except Exception:
        for path in temporary.values():
            path.unlink(missing_ok=True)
        raise
    return manifest


def _evaluation_report(metrics: EvaluationMetrics, warnings: list[str]) -> str:
    warning_lines = "\n".join(f"- {warning}" for warning in warnings) or "- None."
    return f"""# Topic evaluation

- Status: `{metrics.status}`
- Preliminary: {metrics.preliminary}
- Records: {metrics.records}
- Topics: {metrics.topics}
- Original outlier share: {metrics.original_outlier_share:.1%}
- Final outlier share: {metrics.final_outlier_share:.1%}
- Changed-label share: {metrics.changed_label_share:.1%}
- Original silhouette: {metrics.original_geometry.silhouette}
- Final silhouette: {metrics.final_geometry.silhouette}
- Mean bootstrap ARI/NMI: {metrics.mean_bootstrap_ari}/{metrics.mean_bootstrap_nmi}
- Manual annotations: {metrics.manual.annotations}
- Manual topic precision: {metrics.manual.topic_precision}
- Business relevance: {metrics.manual.business_relevance_share}
- Suspicious topic terms: {metrics.suspicious_topic_terms}

## Warnings and failures

{warning_lines}

Automatic metrics cannot produce a final pass without completed manual review and validation. The manual-review sample
contains sensitive local comment text and must remain under ignored data paths. This tracked-safe report contains no
comment text, topic keywords, authors, videos, or source identifiers.
"""
