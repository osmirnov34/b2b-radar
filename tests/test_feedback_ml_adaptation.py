import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from src.analytics import FeedbackMLAdaptationManifest, FeedbackTopicAssignment
from src.operations.feedback_ml_adaptation import adapt_feedback_ml_results


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


def _artifacts(tmp_path: Path, *, record_id: str = "review-1") -> tuple[Path, Path, Path, Path]:
    feedback = tmp_path / "feedback.jsonl"
    row = {
        "feedback_id": "yandex_maps:review-1",
        "source": "yandex_maps",
        "source_record_id": "review-1",
        "text": "Частный текст отзыва",
        "rating": 2,
        "published_on": "2025-04-03",
        "organization_id": "org-1",
        "organization_name": "Стоматология",
        "organization_url": "https://yandex.ru/maps/org/org-1/reviews/",
        "search_query": "стоматология",
    }
    _write_json(feedback, row)
    feedback_manifest = tmp_path / "feedback.manifest.json"
    _write_json(
        feedback_manifest,
        {
            "source": "yandex_maps",
            "source_path": "source.jsonl",
            "source_size_bytes": 1,
            "source_sha256": "a" * 64,
            "output_path": str(feedback),
            "output_size_bytes": feedback.stat().st_size,
            "output_sha256": _sha256(feedback),
            "prepared_at": datetime.now(UTC).isoformat(),
            "records": 1,
            "unique_organizations": 1,
            "rating_distribution": {"2": 1},
            "published_on_min": "2025-04-03",
            "published_on_max": "2025-04-03",
        },
    )
    corpus = tmp_path / "final-corpus.jsonl"
    corpus_id = f"corpus:{hashlib.sha256(record_id.encode()).hexdigest()}"
    _write_json(
        corpus,
        {
            "record_id": record_id,
            "text": "Частный текст отзыва",
            "text_kind": "comment",
            "clean_text": "частный текст отзыва",
            "detected_language": "ru",
            "duplicate_count": 1,
            "corpus_id": corpus_id,
            "cleaned_record_index": 0,
            "semantic_duplicate_count": 0,
        },
    )
    corpus_manifest = tmp_path / "corpus-manifest.json"
    _write_json(
        corpus_manifest,
        {
            "records_path": "clean.jsonl",
            "records_sha256": "b" * 64,
            "cleaning_manifest_path": "cleaning.json",
            "cleaning_manifest_sha256": "c" * 64,
            "embeddings_path": "embeddings.npy",
            "embeddings_sha256": "d" * 64,
            "embedding_manifest_path": "embedding.json",
            "embedding_manifest_sha256": "e" * 64,
            "keep_indices_path": "keep.json",
            "keep_indices_sha256": "f" * 64,
            "groups_path": "groups.jsonl",
            "groups_sha256": "1" * 64,
            "deduplication_manifest_path": "dedup.json",
            "deduplication_manifest_sha256": "2" * 64,
            "corpus_path": str(corpus),
            "corpus_sha256": _sha256(corpus),
            "final_embeddings_path": "final.npy",
            "final_embeddings_sha256": "3" * 64,
            "final_record_ids_path": "ids.jsonl",
            "final_record_ids_sha256": "4" * 64,
            "dimensions": 2,
            "dtype": "float32",
            "stats": {
                "input_records": 1,
                "output_records": 1,
                "removed_semantic_duplicates": 0,
                "output_comments": 1,
                "output_replies": 0,
                "languages": {"ru": 1},
                "unique_videos": 0,
            },
            "created_at": datetime.now(UTC).isoformat(),
        },
    )
    topics = tmp_path / "topic-representations.jsonl"
    _write_json(
        topics,
        {
            "topic_id": 0,
            "name": "ожидание / запись",
            "records": 1,
            "mean_probability": 0.9,
            "languages": {"ru": 1},
            "unique_videos": 0,
            "keywords": [],
            "representative_indices": [0],
        },
    )
    topic_manifest = tmp_path / "topic-manifest.json"
    _write_json(
        topic_manifest,
        {
            "corpus_manifest_path": str(corpus_manifest),
            "corpus_manifest_sha256": _sha256(corpus_manifest),
            "clustering_manifest_path": "clustering.json",
            "clustering_manifest_sha256": "5" * 64,
            "corpus_path": str(corpus),
            "corpus_sha256": _sha256(corpus),
            "embeddings_path": "final.npy",
            "embeddings_sha256": "3" * 64,
            "labels_path": "labels.npy",
            "labels_sha256": "6" * 64,
            "probabilities_path": "probabilities.npy",
            "probabilities_sha256": "7" * 64,
            "input_records": 1,
            "source_topics": 1,
            "topics": 1,
            "omitted_topics": 0,
            "outliers": 0,
            "config": {},
            "stopwords_sha256": "8" * 64,
            "backend": "test",
            "backend_version": "1",
            "representations_path": str(topics),
            "representations_sha256": _sha256(topics),
            "keywords_path": "keywords.jsonl",
            "keywords_sha256": "9" * 64,
            "representative_indices_path": "representatives.jsonl",
            "representative_indices_sha256": "a" * 64,
            "vocabulary_path": "vocabulary.json",
            "vocabulary_sha256": "b" * 64,
            "ctfidf_path": "ctfidf.npz",
            "ctfidf_sha256": "c" * 64,
            "vectorizer_path": "vectorizer.pkl",
            "vectorizer_sha256": "d" * 64,
            "quality": {
                "topics": 1,
                "empty_topics": 0,
                "vocabulary_size": 1,
                "topic_diversity": 1,
                "maximum_keyword_jaccard": 0,
                "similar_topic_pairs": 0,
                "duplicate_names": 0,
                "mean_representative_similarity": 1,
            },
            "warnings": [],
            "created_at": datetime.now(UTC).isoformat(),
        },
    )
    labels = tmp_path / "final-labels.npy"
    confidence = tmp_path / "final-confidence.npy"
    np.save(labels, np.asarray([0], dtype=np.int64), allow_pickle=False)
    np.save(confidence, np.asarray([0.91], dtype=np.float32), allow_pickle=False)
    decisions = tmp_path / "decisions.jsonl"
    _write_json(
        decisions,
        {
            "record_index": 0,
            "final_label": 0,
            "best_topic": None,
            "best_similarity": None,
            "second_topic": None,
            "second_similarity": None,
            "margin": None,
            "reassigned": False,
            "reason": "disabled",
        },
    )
    reassignment_manifest = tmp_path / "reassignment-manifest.json"
    _write_json(
        reassignment_manifest,
        {
            "corpus_manifest_path": str(corpus_manifest),
            "corpus_manifest_sha256": _sha256(corpus_manifest),
            "clustering_manifest_path": "clustering.json",
            "clustering_manifest_sha256": "e" * 64,
            "topic_manifest_path": str(topic_manifest),
            "topic_manifest_sha256": _sha256(topic_manifest),
            "embeddings_path": "final.npy",
            "embeddings_sha256": "3" * 64,
            "source_labels_path": "source-labels.npy",
            "source_labels_sha256": "f" * 64,
            "source_probabilities_path": "source-probabilities.npy",
            "source_probabilities_sha256": "1" * 64,
            "topic_representations_path": str(topics),
            "topic_representations_sha256": _sha256(topics),
            "config": {},
            "eligible_topics": [],
            "ineligible_topics": {},
            "final_labels_path": str(labels),
            "final_labels_sha256": _sha256(labels),
            "final_confidence_path": str(confidence),
            "final_confidence_sha256": _sha256(confidence),
            "decisions_path": str(decisions),
            "decisions_sha256": _sha256(decisions),
            "summary_path": "summary.jsonl",
            "summary_sha256": "2" * 64,
            "metrics": {
                "records": 1,
                "topics": 1,
                "eligible_topics": 0,
                "original_outliers": 0,
                "reassigned_outliers": 0,
                "reassigned_share": 0,
                "remaining_outliers": 0,
                "largest_topic_reassignment_share": 0,
                "mean_accepted_similarity": None,
                "minimum_accepted_similarity": None,
                "mean_accepted_margin": None,
                "rejection_reasons": {"disabled": 1},
            },
            "warnings": [],
            "created_at": datetime.now(UTC).isoformat(),
        },
    )
    return feedback, feedback_manifest, corpus_manifest, reassignment_manifest


def test_adaptation_joins_source_id_without_copying_private_text(tmp_path: Path) -> None:
    feedback, feedback_manifest, corpus_manifest, reassignment_manifest = _artifacts(tmp_path)
    output = tmp_path / "feedback-topics.jsonl"

    result = adapt_feedback_ml_results(
        feedback,
        feedback_manifest,
        corpus_manifest,
        reassignment_manifest,
        output,
    )

    assignment = FeedbackTopicAssignment.model_validate_json(output.read_text(encoding="utf-8"))
    persisted_manifest = FeedbackMLAdaptationManifest.model_validate_json(
        output.with_suffix(".jsonl.manifest.json").read_text(encoding="utf-8"),
    )
    assert assignment.feedback_id == "yandex_maps:review-1"
    assert assignment.topic_id == 0
    assert assignment.topic_name == "ожидание / запись"
    assert assignment.confidence == pytest.approx(0.91)
    assert "Частный текст" not in output.read_text(encoding="utf-8")
    assert result == persisted_manifest
    assert result.unmatched_feedback_records == 0


def test_adaptation_rejects_unknown_corpus_id_without_publishing(tmp_path: Path) -> None:
    feedback, feedback_manifest, corpus_manifest, reassignment_manifest = _artifacts(
        tmp_path,
        record_id="unknown",
    )
    output = tmp_path / "feedback-topics.jsonl"

    with pytest.raises(ValueError, match="no canonical feedback match"):
        adapt_feedback_ml_results(
            feedback,
            feedback_manifest,
            corpus_manifest,
            reassignment_manifest,
            output,
        )

    assert not output.exists()
