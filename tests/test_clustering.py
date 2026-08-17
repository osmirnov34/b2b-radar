import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from src.ml.clustering import (
    ClustererFactory,
    ClusteringManifest,
    ClusterSummary,
    HDBSCANConfig,
    cluster_corpus,
    normalize_cluster_labels,
)
from src.ml.corpus import CorpusManifest, CorpusRecord, CorpusStats
from src.ml.dimensionality_reduction import (
    ReductionArtifactManifest,
    ReductionMode,
    ReductionQuality,
    UMAPConfig,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FakeClusterer:
    library = "fake-hdbscan"
    library_version = "1.0"

    def __init__(
        self,
        labels: list[int],
        probabilities: list[float],
        *,
        relative_validity: float | None = 0.42,
        dbcv: float | None = 0.31,
    ) -> None:
        self.labels = np.asarray(labels, dtype=np.int64)
        self._probabilities = np.asarray(probabilities, dtype=np.float32)
        self._relative_validity = relative_validity
        self._dbcv = dbcv
        self.fit_shape: tuple[int, ...] | None = None

    @property
    def probabilities(self) -> np.ndarray:
        rows = self.fit_shape[0] if self.fit_shape is not None else len(self._probabilities)
        return self._probabilities[:rows]

    @property
    def relative_validity(self) -> float | None:
        return self._relative_validity

    @property
    def dbcv(self) -> float | None:
        return self._dbcv

    def fit_predict(self, vectors: np.ndarray) -> np.ndarray:
        self.fit_shape = vectors.shape
        return self.labels[: len(vectors)]

    def dump(self, path: Path) -> None:
        path.write_bytes(b"trusted fake model")


def _factory(clusterer: FakeClusterer) -> ClustererFactory:
    def create(_config: HDBSCANConfig) -> FakeClusterer:
        return clusterer

    return create


def _build_inputs(
    tmp_path: Path,
    *,
    mode: ReductionMode = ReductionMode.CLUSTERING,
    count: int = 8,
) -> tuple[Path, Path, Path]:
    corpus_path = tmp_path / "final-corpus.jsonl"
    ids_path = tmp_path / "final-record-ids.jsonl"
    records = []
    ids = []
    for index in range(count):
        record_id = f"record-{index}"
        record = CorpusRecord(
            record_id=record_id,
            text=f"Private raw text {index}",
            clean_text=f"Private clean text {index}",
            text_kind="comment" if index % 3 else "reply",
            parent_record_id="parent" if index % 3 == 0 else None,
            detected_language="ru" if index % 2 else "en",
            video_id=f"private-video-{index % 3}",
            corpus_id=f"corpus:{hashlib.sha256(record_id.encode()).hexdigest()}",
            cleaned_record_index=index,
        )
        records.append(record.model_dump_json())
        ids.append(json.dumps(record_id))
    corpus_path.write_text("\n".join(records) + "\n", encoding="utf-8")
    ids_path.write_text("\n".join(ids) + "\n", encoding="utf-8")
    original_embeddings = tmp_path / "final-embeddings.npy"
    np.save(original_embeddings, np.ones((count, 6), dtype=np.float32))
    corpus_manifest_path = tmp_path / "corpus-manifest.json"
    corpus_manifest = CorpusManifest(
        records_path="clean.jsonl",
        records_sha256="0" * 64,
        cleaning_manifest_path="cleaning-manifest.json",
        cleaning_manifest_sha256="1" * 64,
        embeddings_path="embeddings.npy",
        embeddings_sha256="2" * 64,
        embedding_manifest_path="embedding-manifest.json",
        embedding_manifest_sha256="3" * 64,
        keep_indices_path="keep.json",
        keep_indices_sha256="4" * 64,
        groups_path="groups.jsonl",
        groups_sha256="5" * 64,
        deduplication_manifest_path="dedup.json",
        deduplication_manifest_sha256="6" * 64,
        corpus_path=str(corpus_path),
        corpus_sha256=_sha256(corpus_path),
        final_embeddings_path=str(original_embeddings),
        final_embeddings_sha256=_sha256(original_embeddings),
        final_record_ids_path=str(ids_path),
        final_record_ids_sha256=_sha256(ids_path),
        dimensions=6,
        dtype="float32",
        stats=CorpusStats(
            input_records=count,
            output_records=count,
            removed_semantic_duplicates=0,
            output_comments=count - 3,
            output_replies=3,
            languages={"en": 4, "ru": 4},
            unique_videos=3,
        ),
        created_at=datetime.now(UTC),
    )
    corpus_manifest_path.write_text(corpus_manifest.model_dump_json(), encoding="utf-8")
    reduced_path = tmp_path / ("clustering-reduced.npy" if mode == ReductionMode.CLUSTERING else "visualization-2d.npy")
    dimensions = 4 if mode == ReductionMode.CLUSTERING else 2
    generator = np.random.default_rng(9)
    np.save(reduced_path, generator.normal(size=(count, dimensions)).astype(np.float32))
    reduction_manifest_path = tmp_path / "reduction-manifest.json"
    reduction_manifest = ReductionArtifactManifest(
        mode=mode,
        corpus_manifest_path=str(corpus_manifest_path),
        corpus_manifest_sha256=_sha256(corpus_manifest_path),
        input_embeddings_path=str(original_embeddings),
        input_embeddings_sha256=_sha256(original_embeddings),
        input_records=count,
        output_records=count,
        input_dimensions=6,
        output_dimensions=dimensions,
        reduced_path=str(reduced_path),
        reduced_sha256=_sha256(reduced_path),
        model_path="umap-model.pkl",
        model_sha256="7" * 64,
        training_indices_path="training.json",
        training_indices_sha256="8" * 64,
        training_records=count,
        config=UMAPConfig(training_sample_size=None, trustworthiness_sample_size=0),
        effective_n_neighbors=2,
        library="fake-umap",
        library_version="1.0",
        quality=ReductionQuality(
            coordinate_variances=[1.0] * dimensions,
            duplicate_coordinate_share=0,
        ),
        created_at=datetime.now(UTC),
    )
    reduction_manifest_path.write_text(reduction_manifest.model_dump_json(), encoding="utf-8")
    return reduced_path, reduction_manifest_path, corpus_manifest_path


def test_cluster_corpus_normalizes_labels_and_builds_safe_summary(tmp_path: Path) -> None:
    reduced, reduction_manifest, corpus_manifest = _build_inputs(tmp_path)
    fake = FakeClusterer(
        [9, 9, 3, -1, 3, 3, 7, 7],
        [0.8, 0.7, 0.9, 0.0, 0.6, 0.5, 0.95, 0.85],
    )
    output = tmp_path / "clustering"

    manifest = cluster_corpus(
        reduced,
        reduction_manifest,
        corpus_manifest,
        output,
        config=HDBSCANConfig(min_cluster_size=2, minimum_probability=0.6),
        clusterer_factory=_factory(fake),
    )

    labels = np.load(output / "cluster-labels.npy", allow_pickle=False)
    probabilities = np.load(output / "cluster-probabilities.npy", allow_pickle=False)
    summaries = [
        ClusterSummary.model_validate_json(line)
        for line in (output / "cluster-summary.jsonl").read_text().splitlines()
    ]
    assert labels.tolist() == [1, 1, 0, -1, 0, 0, 2, 2]
    assert labels.dtype == np.int64
    assert probabilities.dtype == np.float32
    assert manifest.label_mapping == {3: 0, 9: 1, 7: 2}
    assert manifest.metrics.clusters == 3
    assert manifest.metrics.outliers == 1
    assert manifest.metrics.low_confidence_records == 2
    assert summaries[0].records == 3
    assert summaries[0].minimum_record_index == 2
    assert fake.fit_shape == (8, 4)
    assert ClusteringManifest.model_validate_json((output / "clustering-manifest.json").read_text()) == manifest
    safe_text = (output / "clustering-report.md").read_text() + (output / "cluster-summary.jsonl").read_text()
    assert "Private" not in safe_text
    assert "private-video" not in safe_text


def test_cluster_corpus_rejects_visualization_space(tmp_path: Path) -> None:
    reduced, reduction_manifest, corpus_manifest = _build_inputs(tmp_path, mode=ReductionMode.VISUALIZATION)
    with pytest.raises(ValueError, match="requires a clustering reduction manifest"):
        cluster_corpus(
            reduced,
            reduction_manifest,
            corpus_manifest,
            tmp_path / "output",
            clusterer_factory=_factory(FakeClusterer([-1] * 8, [0.0] * 8)),
        )


def test_cluster_corpus_rejects_tampered_reduction(tmp_path: Path) -> None:
    reduced, reduction_manifest, corpus_manifest = _build_inputs(tmp_path)
    with reduced.open("ab") as target:
        target.write(b"tampered")
    with pytest.raises(ValueError, match="checksum does not match"):
        cluster_corpus(
            reduced,
            reduction_manifest,
            corpus_manifest,
            tmp_path / "output",
            clusterer_factory=_factory(FakeClusterer([-1] * 8, [0.0] * 8)),
        )


@pytest.mark.parametrize(
    ("labels", "probabilities", "message"),
    [
        ([0, 0], [0.5] * 8, "labels shape"),
        ([0] * 8, [0.5, 1.2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], "finite values in"),
        ([0] * 8, [0.5, float("nan"), 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], "finite values in"),
    ],
)
def test_cluster_corpus_rejects_invalid_clusterer_results(
    tmp_path: Path,
    labels: list[int],
    probabilities: list[float],
    message: str,
) -> None:
    reduced, reduction_manifest, corpus_manifest = _build_inputs(tmp_path)
    with pytest.raises(ValueError, match=message):
        cluster_corpus(
            reduced,
            reduction_manifest,
            corpus_manifest,
            tmp_path / "output",
            clusterer_factory=_factory(FakeClusterer(labels, probabilities)),
        )


def test_cluster_corpus_supports_all_outliers_and_warns(tmp_path: Path) -> None:
    reduced, reduction_manifest, corpus_manifest = _build_inputs(tmp_path)
    manifest = cluster_corpus(
        reduced,
        reduction_manifest,
        corpus_manifest,
        tmp_path / "output",
        clusterer_factory=_factory(FakeClusterer([-1] * 8, [0.0] * 8, relative_validity=None, dbcv=None)),
    )
    assert manifest.metrics.clusters == 0
    assert manifest.metrics.outlier_share == 1
    assert any("only outliers" in warning for warning in manifest.warnings)
    assert (tmp_path / "output" / "cluster-summary.jsonl").read_text() == ""


def test_cluster_corpus_protects_artifacts_and_supports_limit(tmp_path: Path) -> None:
    reduced, reduction_manifest, corpus_manifest = _build_inputs(tmp_path)
    output = tmp_path / "output"
    fake = FakeClusterer([0] * 8, [0.9] * 8)
    kwargs = {"clusterer_factory": _factory(fake), "limit": 5}
    manifest = cluster_corpus(reduced, reduction_manifest, corpus_manifest, output, **kwargs)
    assert manifest.input_records == 8
    assert manifest.output_records == 5
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        cluster_corpus(reduced, reduction_manifest, corpus_manifest, output, **kwargs)
    replaced = cluster_corpus(reduced, reduction_manifest, corpus_manifest, output, force=True, **kwargs)
    assert replaced.output_records == 5


def test_normalize_cluster_labels_is_deterministic_and_rejects_invalid_values() -> None:
    normalized, mapping = normalize_cluster_labels(np.asarray([5, 2, 5, 2, -1, 9], dtype=np.int32))
    assert normalized.tolist() == [0, 1, 0, 1, -1, 2]
    assert mapping == {5: 0, 2: 1, 9: 2}
    with pytest.raises(ValueError, match="only use -1"):
        normalize_cluster_labels(np.asarray([0, -2], dtype=np.int64))
