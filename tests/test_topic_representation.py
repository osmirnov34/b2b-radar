import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from src.ml.clustering import ClusteringManifest, ClusteringMetrics, HDBSCANConfig
from src.ml.corpus import CorpusManifest, CorpusRecord, CorpusStats
from src.ml.topic_representation import (
    TopicBackendFactory,
    TopicRepresentation,
    TopicRepresentationConfig,
    TopicRepresentationManifest,
    build_topic_representations,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FakeTopicBackend:
    name = "fake-ctfidf"
    library_version = "1.0"
    vocabulary = ("automation", "crm", "sales", "support")

    def __init__(self) -> None:
        self.documents: list[str] = []

    def fit(self, documents: list[str]) -> None:
        self.documents = list(documents)

    def top_terms(self, topic_row: int, count: int) -> list[tuple[str, float]]:
        terms = (
            [("crm", 0.9), ("automation", 0.7), ("sales", 0.5)],
            [("support", 0.8), ("automation", 0.6)],
        )
        return terms[topic_row][:count]

    def dump_vectorizer(self, path: Path) -> None:
        path.write_bytes(b"trusted fake vectorizer")

    def dump_matrix(self, path: Path) -> None:
        path.write_bytes(b"fake sparse matrix")


def _factory(backend: FakeTopicBackend) -> TopicBackendFactory:
    def create(_config: TopicRepresentationConfig, _stopwords: set[str]) -> FakeTopicBackend:
        return backend

    return create


def _save_array(path: Path, values: np.ndarray) -> None:
    with path.open("wb") as target:
        np.save(target, values, allow_pickle=False)


def _build_inputs(
    tmp_path: Path,
    *,
    labels_values: list[int] | None = None,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    labels_values = labels_values or [0, 0, 0, 1, 1, -1, -1]
    count = len(labels_values)
    corpus_path = tmp_path / "final-corpus.jsonl"
    ids_path = tmp_path / "final-record-ids.jsonl"
    records = []
    ids = []
    texts = [
        "CRM automation sales",
        "CRM pipeline automation",
        "Sales CRM integration",
        "Customer support workflow",
        "Support automation desk",
        "Private outlier phrase one",
        "Private outlier phrase two",
    ][:count]
    for index, text in enumerate(texts):
        record_id = f"record-{index}"
        record = CorpusRecord(
            record_id=record_id,
            text=text,
            clean_text=text,
            text_kind="comment",
            detected_language="en" if index % 2 else "ru",
            video_id=f"video-{index % 4}",
            author=f"author-{index % 5}",
            corpus_id=f"corpus:{hashlib.sha256(record_id.encode()).hexdigest()}",
            cleaned_record_index=index,
        )
        records.append(record.model_dump_json())
        ids.append(json.dumps(record_id))
    corpus_path.write_text("\n".join(records) + "\n", encoding="utf-8")
    ids_path.write_text("\n".join(ids) + "\n", encoding="utf-8")
    embeddings = np.asarray(
        [[1.0, index / 20, 0.1] if index < 3 else [0.1, 1.0, index / 20] for index in range(count)],
        dtype=np.float32,
    )
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_path = tmp_path / "final-embeddings.npy"
    _save_array(embeddings_path, embeddings)
    corpus_manifest_path = tmp_path / "corpus-manifest.json"
    corpus_manifest = CorpusManifest(
        records_path="clean.jsonl",
        records_sha256="0" * 64,
        cleaning_manifest_path="cleaning.json",
        cleaning_manifest_sha256="1" * 64,
        embeddings_path="source-embeddings.npy",
        embeddings_sha256="2" * 64,
        embedding_manifest_path="embedding.json",
        embedding_manifest_sha256="3" * 64,
        keep_indices_path="keep.json",
        keep_indices_sha256="4" * 64,
        groups_path="groups.jsonl",
        groups_sha256="5" * 64,
        deduplication_manifest_path="dedup.json",
        deduplication_manifest_sha256="6" * 64,
        corpus_path=str(corpus_path),
        corpus_sha256=_sha256(corpus_path),
        final_embeddings_path=str(embeddings_path),
        final_embeddings_sha256=_sha256(embeddings_path),
        final_record_ids_path=str(ids_path),
        final_record_ids_sha256=_sha256(ids_path),
        dimensions=3,
        dtype="float32",
        stats=CorpusStats(
            input_records=count,
            output_records=count,
            removed_semantic_duplicates=0,
            output_comments=count,
            output_replies=0,
            languages={"en": count // 2, "ru": count - count // 2},
            unique_videos=4,
        ),
        created_at=datetime.now(UTC),
    )
    corpus_manifest_path.write_text(corpus_manifest.model_dump_json(), encoding="utf-8")
    labels_path = tmp_path / "cluster-labels.npy"
    probabilities_path = tmp_path / "cluster-probabilities.npy"
    _save_array(labels_path, np.asarray(labels_values, dtype=np.int64))
    probabilities = np.asarray([0.95, 0.85, 0.75, 0.9, 0.8, 0.0, 0.0][:count], dtype=np.float32)
    _save_array(probabilities_path, probabilities)
    summary_path = tmp_path / "cluster-summary.jsonl"
    summary_path.write_text("", encoding="utf-8")
    clustering_manifest_path = tmp_path / "clustering-manifest.json"
    non_outlier = [label for label in labels_values if label >= 0]
    cluster_count = len(set(non_outlier))
    outliers = labels_values.count(-1)
    clustering_manifest = ClusteringManifest(
        reduction_manifest_path="reduction.json",
        reduction_manifest_sha256="7" * 64,
        reduced_path="reduced.npy",
        reduced_sha256="8" * 64,
        corpus_manifest_path=str(corpus_manifest_path),
        corpus_manifest_sha256=_sha256(corpus_manifest_path),
        corpus_sha256=_sha256(corpus_path),
        input_records=count,
        output_records=count,
        input_dimensions=5,
        config=HDBSCANConfig(min_cluster_size=2),
        label_mapping={index: index for index in range(cluster_count)},
        labels_path=str(labels_path),
        labels_sha256=_sha256(labels_path),
        probabilities_path=str(probabilities_path),
        probabilities_sha256=_sha256(probabilities_path),
        summary_path=str(summary_path),
        summary_sha256=_sha256(summary_path),
        model_path="hdbscan.pkl",
        model_sha256="9" * 64,
        library="fake-hdbscan",
        library_version="1.0",
        metrics=ClusteringMetrics(
            records=count,
            clusters=cluster_count,
            outliers=outliers,
            outlier_share=outliers / count,
            smallest_cluster=min(Counter(non_outlier).values(), default=None),
            largest_cluster=max(Counter(non_outlier).values(), default=None),
            median_cluster_size=2.5 if cluster_count else None,
            dominant_cluster_share=3 / count if cluster_count else 0,
            micro_clusters=cluster_count,
            micro_cluster_share=1 if cluster_count else 0,
            mean_probability=float(np.mean(probabilities)),
            low_confidence_records=outliers,
            low_confidence_share=outliers / count,
        ),
        warnings=[],
        created_at=datetime.now(UTC),
    )
    clustering_manifest_path.write_text(clustering_manifest.model_dump_json(), encoding="utf-8")
    return (
        corpus_path,
        embeddings_path,
        labels_path,
        probabilities_path,
        corpus_manifest_path,
        clustering_manifest_path,
    )


def _run(
    paths: tuple[Path, ...],
    output: Path,
    backend: FakeTopicBackend,
    **kwargs: object,
) -> TopicRepresentationManifest:
    return build_topic_representations(
        *paths,
        output,
        config=TopicRepresentationConfig(
            representatives_per_topic=2,
            minimum_representative_probability=0.5,
            similar_topic_jaccard_warning=0.2,
        ),
        backend_factory=_factory(backend),
        stopwords={"and", "the"},
        **kwargs,
    )


def test_build_topic_representations_preserves_labels_and_excludes_outliers(tmp_path: Path) -> None:
    paths = _build_inputs(tmp_path)
    backend = FakeTopicBackend()
    output = tmp_path / "topics"

    manifest = _run(paths, output, backend)

    representations = [
        TopicRepresentation.model_validate_json(line)
        for line in (output / "topic-representations.jsonl").read_text().splitlines()
    ]
    assert len(backend.documents) == 2
    assert all("outlier" not in document.casefold() for document in backend.documents)
    assert [item.name for item in representations] == ["crm / automation / sales", "support / automation"]
    assert [item.records for item in representations] == [3, 2]
    assert all(len(item.representative_indices) == 2 for item in representations)
    assert manifest.topics == 2
    assert manifest.outliers == 2
    assert manifest.quality.vocabulary_size == 4
    assert manifest.quality.similar_topic_pairs == 1
    assert TopicRepresentationManifest.model_validate_json(
        (output / "topic-representation-manifest.json").read_text(),
    ) == manifest
    report = (output / "topic-representation-report.md").read_text()
    assert "Private outlier" not in report
    assert "CRM automation" not in report


def test_topic_limit_builds_prefix_topics_without_changing_source_labels(tmp_path: Path) -> None:
    paths = _build_inputs(tmp_path)
    original_labels = paths[2].read_bytes()
    manifest = _run(paths, tmp_path / "topics", FakeTopicBackend(), limit_topics=1)
    assert manifest.source_topics == 2
    assert manifest.topics == 1
    assert manifest.omitted_topics == 1
    assert paths[2].read_bytes() == original_labels


def test_topic_representation_rejects_noncontiguous_labels_and_tampering(tmp_path: Path) -> None:
    paths = _build_inputs(tmp_path, labels_values=[0, 0, 2, 2, -1, -1, -1])
    with pytest.raises(ValueError, match="normalized contiguous"):
        _run(paths, tmp_path / "invalid", FakeTopicBackend())

    paths = _build_inputs(tmp_path / "tampered")
    with paths[1].open("ab") as target:
        target.write(b"tampered")
    with pytest.raises(ValueError, match="embeddings checksum"):
        _run(paths, tmp_path / "tampered-output", FakeTopicBackend())


def test_topic_representation_handles_all_outliers(tmp_path: Path) -> None:
    paths = _build_inputs(tmp_path, labels_values=[-1] * 7)
    backend = FakeTopicBackend()
    manifest = _run(paths, tmp_path / "topics", backend)
    assert backend.documents == []
    assert manifest.topics == 0
    assert manifest.outliers == 7
    assert manifest.quality.empty_topics == 0
    assert (tmp_path / "topics" / "topic-representations.jsonl").read_text() == ""


def test_topic_representation_protects_existing_outputs(tmp_path: Path) -> None:
    paths = _build_inputs(tmp_path)
    output = tmp_path / "topics"
    _run(paths, output, FakeTopicBackend())
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _run(paths, output, FakeTopicBackend())
    assert _run(paths, output, FakeTopicBackend(), force=True).topics == 2
