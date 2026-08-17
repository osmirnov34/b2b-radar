import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from src.ml.corpus import CorpusManifest, CorpusRecord, CorpusStats
from src.ml.dimensionality_reduction import (
    ReducerFactory,
    ReductionArtifactManifest,
    ReductionMode,
    UMAPConfig,
    deterministic_training_indices,
    reduce_dimensions,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FakeReducer:
    library = "fake-reducer"
    library_version = "1.0"

    def __init__(self, components: int, calls: list[tuple[str, int]]) -> None:
        self.components = components
        self.calls = calls

    def fit(self, vectors: np.ndarray) -> None:
        self.calls.append(("fit", len(vectors)))

    def transform(self, vectors: np.ndarray) -> np.ndarray:
        self.calls.append(("transform", len(vectors)))
        return np.asarray(vectors[:, : self.components] + 0.25, dtype=np.float32)

    def dump(self, path: Path) -> None:
        path.write_bytes(f"fake:{self.components}".encode())


def _factory(calls: list[tuple[str, int]]) -> ReducerFactory:
    def create(
        _mode: ReductionMode,
        components: int,
        _neighbors: int,
        _min_dist: float,
        _config: UMAPConfig,
    ) -> FakeReducer:
        return FakeReducer(components, calls)

    return create


def _build_corpus(tmp_path: Path, count: int = 9, dimensions: int = 6) -> tuple[Path, Path, np.ndarray]:
    corpus_path = tmp_path / "final-corpus.jsonl"
    ids_path = tmp_path / "final-record-ids.jsonl"
    rows = []
    ids = []
    for index in range(count):
        record = CorpusRecord(
            record_id=f"record-{index}",
            text=f"Private text {index}",
            clean_text=f"Private clean text {index}",
            text_kind="comment",
            detected_language="en",
            corpus_id=f"corpus:{hashlib.sha256(f'record-{index}'.encode()).hexdigest()}",
            cleaned_record_index=index,
        )
        rows.append(record.model_dump_json())
        ids.append(json.dumps(record.record_id))
    corpus_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    ids_path.write_text("\n".join(ids) + "\n", encoding="utf-8")
    generator = np.random.default_rng(7)
    embeddings = generator.normal(size=(count, dimensions)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_path = tmp_path / "final-embeddings.npy"
    np.save(embeddings_path, embeddings)
    manifest_path = tmp_path / "corpus-manifest.json"
    manifest = CorpusManifest(
        records_path="clean.jsonl",
        records_sha256="0" * 64,
        cleaning_manifest_path="cleaning-manifest.json",
        cleaning_manifest_sha256="1" * 64,
        embeddings_path="embeddings.npy",
        embeddings_sha256="2" * 64,
        embedding_manifest_path="embedding-manifest.json",
        embedding_manifest_sha256="3" * 64,
        keep_indices_path="keep-indices.json",
        keep_indices_sha256="4" * 64,
        groups_path="groups.jsonl",
        groups_sha256="5" * 64,
        deduplication_manifest_path="deduplication-manifest.json",
        deduplication_manifest_sha256="6" * 64,
        corpus_path=str(corpus_path),
        corpus_sha256=_sha256(corpus_path),
        final_embeddings_path=str(embeddings_path),
        final_embeddings_sha256=_sha256(embeddings_path),
        final_record_ids_path=str(ids_path),
        final_record_ids_sha256=_sha256(ids_path),
        dimensions=dimensions,
        dtype="float32",
        stats=CorpusStats(
            input_records=count,
            output_records=count,
            removed_semantic_duplicates=0,
            output_comments=count,
            output_replies=0,
            languages={"en": count},
            unique_videos=0,
        ),
        created_at=datetime.now(UTC),
    )
    manifest_path.write_text(manifest.model_dump_json(), encoding="utf-8")
    return embeddings_path, manifest_path, embeddings


def test_reduce_dimensions_creates_separate_aligned_spaces_in_batches(tmp_path: Path) -> None:
    embeddings_path, manifest_path, embeddings = _build_corpus(tmp_path)
    calls: list[tuple[str, int]] = []
    output = tmp_path / "umap"
    config = UMAPConfig(
        clustering_components=4,
        visualization_components=2,
        training_sample_size=5,
        transform_batch_size=4,
        trustworthiness_sample_size=0,
    )

    manifests = reduce_dimensions(
        embeddings_path,
        manifest_path,
        output,
        config=config,
        reducer_factory=_factory(calls),
    )

    clustering = np.load(output / "clustering-reduced.npy", allow_pickle=False)
    visualization = np.load(output / "visualization-2d.npy", allow_pickle=False)
    assert clustering.shape == (9, 4)
    assert visualization.shape == (9, 2)
    assert np.allclose(clustering, embeddings[:, :4] + 0.25)
    assert np.allclose(visualization, embeddings[:, :2] + 0.25)
    assert calls.count(("fit", 5)) == 2
    assert calls.count(("transform", 4)) == 4
    assert calls.count(("transform", 1)) == 2
    for mode in ReductionMode:
        persisted = ReductionArtifactManifest.model_validate_json(
            (output / f"{mode.value}-manifest.json").read_text(),
        )
        assert persisted == manifests[mode]
        assert persisted.training_records == 5
        assert persisted.library == "fake-reducer"
        assert persisted.quality.trustworthiness is None


def test_training_sample_is_deterministic_sorted_and_seeded() -> None:
    first = deterministic_training_indices(100, 12, 42)
    assert first == sorted(first)
    assert len(first) == len(set(first)) == 12
    assert first == deterministic_training_indices(100, 12, 42)
    assert first != deterministic_training_indices(100, 12, 43)
    assert deterministic_training_indices(5, None, 42) == [0, 1, 2, 3, 4]


def test_reduce_dimensions_rejects_bad_output_and_tampered_input(tmp_path: Path) -> None:
    embeddings_path, manifest_path, _embeddings = _build_corpus(tmp_path)

    class BadReducer(FakeReducer):
        def transform(self, vectors: np.ndarray) -> np.ndarray:
            return np.full((len(vectors), self.components + 1), np.nan, dtype=np.float32)

    def bad_factory(
        _mode: ReductionMode,
        components: int,
        _neighbors: int,
        _min_dist: float,
        _config: UMAPConfig,
    ) -> BadReducer:
        return BadReducer(components, [])

    with pytest.raises(ValueError, match="reducer returned shape"):
        reduce_dimensions(
            embeddings_path,
            manifest_path,
            tmp_path / "bad-output",
            config=UMAPConfig(trustworthiness_sample_size=0),
            modes=(ReductionMode.CLUSTERING,),
            reducer_factory=bad_factory,
        )

    with embeddings_path.open("ab") as target:
        target.write(b"tampered")
    with pytest.raises(ValueError, match="checksum does not match"):
        reduce_dimensions(
            embeddings_path,
            manifest_path,
            tmp_path / "tampered-output",
            reducer_factory=_factory([]),
        )


def test_reduce_dimensions_protects_existing_artifacts_and_supports_limit(tmp_path: Path) -> None:
    embeddings_path, manifest_path, _embeddings = _build_corpus(tmp_path)
    output = tmp_path / "umap"
    config = UMAPConfig(training_sample_size=4, transform_batch_size=2, trustworthiness_sample_size=0)
    kwargs = {
        "config": config,
        "modes": (ReductionMode.VISUALIZATION,),
        "reducer_factory": _factory([]),
        "limit": 6,
    }
    result = reduce_dimensions(embeddings_path, manifest_path, output, **kwargs)
    assert result[ReductionMode.VISUALIZATION].input_records == 9
    assert result[ReductionMode.VISUALIZATION].output_records == 6
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        reduce_dimensions(embeddings_path, manifest_path, output, **kwargs)
    replaced = reduce_dimensions(embeddings_path, manifest_path, output, force=True, **kwargs)
    assert replaced[ReductionMode.VISUALIZATION].output_records == 6
