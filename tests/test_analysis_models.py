from pathlib import Path

import pytest
from pydantic import ValidationError

from src.analysis import (
    AnalysisConfig,
    AnalysisCounts,
    CleaningDecision,
    CleaningReason,
    CleaningResult,
    CleaningStats,
    CommentRecord,
    DeduplicationResult,
    DeduplicationStats,
    DuplicateGroup,
    DuplicatePair,
    ExportedComment,
    TopicAssignment,
)


def test_comment_record_converts_export_and_preserves_provenance() -> None:
    exported = ExportedComment(
        comment_text="  Нужен простой CRM  ",
        comment_author="Анна",
        video_channel="Бизнес",
        search_query="crm",
        video_id="video-1",
        video_title="Автоматизация",
        video_url="https://example.test/video-1",
    )

    comment = CommentRecord.from_export(exported)
    cluster_comment = comment.to_cluster_comment()

    assert comment.text == "Нужен простой CRM"
    assert comment.video_url == "https://example.test/video-1"
    assert cluster_comment.author == "Анна"
    assert cluster_comment.channel == "Бизнес"
    assert cluster_comment.video_url == comment.video_url


def test_comment_record_builds_youtube_url_and_normalized_key() -> None:
    comment = CommentRecord(text="  CRM   ДЛЯ бизнеса ", video_id="abc")

    assert comment.video_url == "https://www.youtube.com/watch?v=abc"
    assert comment.normalized_text_key == "crm для бизнеса"


def test_internal_models_are_frozen_and_forbid_extra_fields() -> None:
    comment = CommentRecord(text="Text")

    with pytest.raises(ValidationError, match="frozen"):
        comment.text = "Changed"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        CommentRecord.model_validate({"text": "Text", "unknown": True})


@pytest.mark.parametrize(
    ("decision", "message"),
    [
        ({"keep": True, "reason": CleaningReason.EMPTY}, "kept comments cannot have"),
        ({"keep": False}, "removed comments require"),
    ],
)
def test_cleaning_decision_requires_consistent_reason(decision: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        CleaningDecision.model_validate(decision)


def test_cleaning_result_validates_stats() -> None:
    stats = CleaningStats(
        n_input=2,
        n_kept=1,
        removed_by_reason={CleaningReason.TOO_SHORT: 1},
    )

    result = CleaningResult(comments=[CommentRecord(text="Kept")], stats=stats)

    assert result.stats.n_kept == 1
    with pytest.raises(ValidationError, match="cleaning counts must add up"):
        CleaningStats(n_input=3, n_kept=1, removed_by_reason={CleaningReason.EMPTY: 1})


def test_duplicate_models_validate_indices_and_counts() -> None:
    pair = DuplicatePair(representative_index=0, duplicate_index=1, similarity=0.96)
    group = DuplicateGroup(representative_index=0, duplicate_indices=[1, 2])
    stats = DeduplicationStats(n_input=3, n_kept=1, n_removed=2, threshold=0.95)
    result = DeduplicationResult(keep_indices=[0], sample_pairs=[pair], groups=[group], stats=stats)

    assert result.stats.n_removed == 2
    with pytest.raises(ValidationError, match="must differ"):
        DuplicatePair(representative_index=1, duplicate_index=1)
    with pytest.raises(ValidationError, match="must be unique"):
        DuplicateGroup(representative_index=0, duplicate_indices=[1, 1])


def test_topic_assignment_allows_only_outlier_or_non_negative_topic() -> None:
    assert TopicAssignment(comment_index=0, topic_id=-1).topic_id == -1
    with pytest.raises(ValidationError):
        TopicAssignment(comment_index=0, topic_id=-2)


def test_analysis_counts_validate_processing_order() -> None:
    counts = AnalysisCounts(
        n_input=100,
        n_after_clean=80,
        n_after_dedup=70,
        n_topics=5,
        n_outliers_before_reduction=20,
        n_outliers=10,
    )

    assert counts.n_after_dedup == 70
    with pytest.raises(ValidationError, match="n_after_dedup cannot exceed"):
        AnalysisCounts(
            n_input=100,
            n_after_clean=80,
            n_after_dedup=90,
            n_topics=5,
            n_outliers_before_reduction=20,
            n_outliers=10,
        )


def test_analysis_config_can_be_embedded_in_internal_result_models() -> None:
    config = AnalysisConfig(input_path=Path("comments.jsonl"))

    assert config.input_path == Path("comments.jsonl")
