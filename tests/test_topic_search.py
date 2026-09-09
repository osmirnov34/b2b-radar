import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from src.ml.corpus import CorpusManifest, CorpusStats
from src.ml.reporting import AnalysisArtifacts, AnalysisSummary
from src.ml.schemas import EmbeddingArtifactManifest
from src.ml.topic_representation import TopicKeyword, TopicRepresentation, TopicRepresentationManifest
from src.ml.topic_search import TopicSearchConfig, search_user_topic, write_topic_search_result

if TYPE_CHECKING:
    from src.ml.clustering import ClusteringManifest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _Encoder:
    device = "cpu"

    def __init__(self, vector: list[float] | None = None) -> None:
        self.vector = vector or [1.0, 0.0]

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        assert texts == ["delivery failures"]
        return np.asarray([self.vector], dtype=np.float32)


def _artifacts(tmp_path: Path) -> AnalysisArtifacts:
    run = tmp_path / "ml-runs/run-1"
    corpus_path = run / "07-corpus/final-corpus.jsonl"
    matrix_path = run / "07-corpus/final-embeddings.npy"
    embedding_manifest_path = run / "05-embeddings/embedding-manifest.json"
    corpus_path.parent.mkdir(parents=True)
    embedding_manifest_path.parent.mkdir(parents=True)
    rows = [
        {
            "corpus_id": f"corpus:{index:064x}",
            "cleaned_record_index": index,
            "record_id": f"record-{index}",
            "text": text,
            "clean_text": text,
            "text_kind": "comment",
            "author": f"private-author-{index}",
            "detected_language": "en",
            "video_id": f"video{index}",
            "video_title": f"Video {index}",
            "video_channel": "Channel",
            "search_query": "private query",
        }
        for index, text in enumerate(
            (
                "Delivery does not work and refund is delayed",
                "Delivery workflow overview",
                "Accounting tutorial",
                "Delivery complaint from another segment",
            )
        )
    ]
    corpus_path.write_text("".join(f"{json.dumps(row)}\n" for row in rows), encoding="utf-8")
    np.save(
        matrix_path,
        np.asarray([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0], [0.9, 0.43589]], dtype=np.float32),
        allow_pickle=False,
    )
    embedding_manifest = EmbeddingArtifactManifest(
        records_sha256="b" * 64,
        embeddings_sha256="c" * 64,
        n_records=4,
        dimensions=2,
        model_name="test/model",
        model_revision="revision-1",
        normalized=True,
        max_seq_length=128,
        prompt_prefix="query: ",
    )
    embedding_manifest_path.write_text(f"{embedding_manifest.model_dump_json()}\n", encoding="utf-8")
    corpus = CorpusManifest.model_construct(
        embedding_manifest_path=str(embedding_manifest_path),
        embedding_manifest_sha256=_sha256(embedding_manifest_path),
        corpus_path=str(corpus_path),
        corpus_sha256=_sha256(corpus_path),
        final_embeddings_path=str(matrix_path),
        final_embeddings_sha256=_sha256(matrix_path),
        dimensions=2,
        stats=CorpusStats(
            input_records=4,
            output_records=4,
            removed_semantic_duplicates=0,
            output_comments=4,
            output_replies=0,
            languages={"en": 4},
            unique_videos=4,
        ),
    )
    topics = tuple(
        TopicRepresentation(
            topic_id=topic_id,
            name=name,
            records=records,
            mean_probability=0.8,
            languages={"en": records},
            unique_videos=records,
            keywords=[TopicKeyword(term=name, weight=1.0, rank=1, kind="word")],
            representative_indices=[topic_id],
        )
        for topic_id, name, records in ((0, "delivery", 2), (1, "accounting", 1))
    )
    return AnalysisArtifacts(
        run_dir=run,
        run_id="run-1",
        pipeline_schema_version=1,
        pipeline_manifest_sha256="a" * 64,
        corpus_path=corpus_path,
        coordinates=np.zeros((4, 2), dtype=np.float32),
        labels=np.asarray([0, 0, 1, -1], dtype=np.int64),
        confidence=np.asarray([0.9, 0.8, 0.7, 0.0], dtype=np.float32),
        corpus=corpus,
        clustering=cast("ClusteringManifest", None),
        topics_manifest=cast("TopicRepresentationManifest", None),
        topics=topics,
        reassignment=None,
        summary=AnalysisSummary(
            records=4,
            topics=2,
            outliers=1,
            outlier_share=0.25,
            mean_confidence=0.6,
            largest_topic_share=0.5,
            pipeline_status="awaiting_review",
            warnings=[],
        ),
    )


def test_topic_search_finds_subcorpus_problem_evidence_and_omits_authors(tmp_path: Path) -> None:
    result = search_user_topic(
        _artifacts(tmp_path),
        "  delivery   failures ",
        config=TopicSearchConfig(minimum_similarity=0.5, maximum_matches=10),
        encoder=_Encoder(),
        searched_at=datetime(2026, 9, 1, tzinfo=UTC),
    )

    assert result.query == "delivery failures"
    assert result.relevant_records == 2
    assert result.problem_records == 1
    assert len(result.groups) == 1
    assert result.groups[0].topic_name == "delivery"
    assert result.groups[0].problem_share == 0.5
    assert result.groups[0].evidence[0].problem_signals
    assert result.groups[0].evidence[0].video_url.startswith("https://www.youtube.com/watch?v=")
    assert "author" not in result.model_dump_json()
    assert result.model_name == "test/model"
    assert result.model_revision == "revision-1"


def test_topic_search_outliers_require_explicit_opt_in(tmp_path: Path) -> None:
    result = search_user_topic(
        _artifacts(tmp_path),
        "delivery failures",
        config=TopicSearchConfig(minimum_similarity=0.85, include_outliers=True),
        encoder=_Encoder(),
        searched_at=datetime(2026, 9, 1, tzinfo=UTC),
    )

    assert result.relevant_records == 2
    assert {group.topic_id for group in result.groups} == {0, None}
    assert next(group for group in result.groups if group.topic_id is None).topic_name == "Outliers"


def test_topic_search_validates_query_encoder_and_embedding_checksum(tmp_path: Path) -> None:
    artifacts = _artifacts(tmp_path)
    with pytest.raises(ValueError, match="query must contain"):
        search_user_topic(artifacts, " ", encoder=_Encoder())
    with pytest.raises(ValueError, match="invalid shape"):
        search_user_topic(artifacts, "delivery failures", encoder=_Encoder([1.0]))

    Path(artifacts.corpus.final_embeddings_path).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        search_user_topic(artifacts, "delivery failures", encoder=_Encoder())


def test_topic_search_export_is_restricted_atomic_and_outside_run(tmp_path: Path) -> None:
    artifacts = _artifacts(tmp_path)
    result = search_user_topic(
        artifacts,
        "delivery failures",
        encoder=_Encoder(),
        searched_at=datetime(2026, 9, 1, tzinfo=UTC),
    )
    output = tmp_path / "topic-searches/search-1"
    manifest = write_topic_search_result(result, output, artifacts.run_dir)

    assert manifest.classification == "restricted_user_topic_search"
    assert manifest.result_sha256 == _sha256(output / "topic-search-result.json")
    assert manifest.query_sha256 == hashlib.sha256(result.query.encode()).hexdigest()
    with pytest.raises(FileExistsError, match="already exists"):
        write_topic_search_result(result, output, artifacts.run_dir)
    with pytest.raises(ValueError, match="outside the immutable run"):
        write_topic_search_result(result, artifacts.run_dir / "search", artifacts.run_dir)
