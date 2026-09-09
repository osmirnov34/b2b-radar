import hashlib
import inspect
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.ml import reporting
from src.ml.cleaning_dataset import DatasetCleaningManifest, DatasetCleaningStats
from src.ml.clustering import ClusteringManifest, ClusterSummary, HDBSCANConfig
from src.ml.corpus import CorpusManifest, CorpusStats
from src.ml.inspection import DatasetInspection
from src.ml.models import CleaningReason, DeduplicationStats
from src.ml.outlier_reassignment import (
    FinalClusterSummary,
    OutlierDecision,
    OutlierDecisionReason,
    OutlierReassignmentConfig,
    OutlierReassignmentManifest,
    OutlierReassignmentMetrics,
)
from src.ml.reporting import (
    AnalysisArtifacts,
    AnalysisSummary,
    AssignmentExplanation,
    AssignmentOutcome,
    AssignmentReviewComment,
    AssignmentReviewKind,
    ClusterCard,
    DataScope,
    ProblemAssessmentStatus,
    ProblemPriorityConfig,
    ProblemSignalConfig,
    ReassignmentStatus,
    ReportManifest,
    RepresentativeComment,
    TopicSummaryRow,
    assess_topic_problem_signals,
    assignment_review_comments,
    build_cluster_cards,
    build_data_lineage,
    explain_assignment,
    get_cluster_card,
    processing_flow,
    rank_problem_topics,
    representative_comments,
    stratified_plot_indices,
    topic_summary_rows,
    write_analysis_tables,
    write_problem_priority_report,
    write_problem_signal_report,
)
from src.ml.semantic_deduplication import SemanticDeduplicationManifest
from src.ml.splitting import DatasetSplitManifest, SplitName, SplitStats
from src.ml.topic_representation import (
    RepresentativeIndices,
    TopicKeyword,
    TopicRepresentation,
    TopicRepresentationManifest,
)


def _artifacts(tmp_path: Path) -> AnalysisArtifacts:
    topics = (
        TopicRepresentation(
            topic_id=0,
            name="delivery / order",
            records=3,
            mean_probability=0.8,
            languages={"ru": 3},
            unique_videos=2,
            keywords=[TopicKeyword(term="delivery", weight=0.7, rank=1, kind="word")],
            representative_indices=[0],
        ),
    )
    return AnalysisArtifacts(
        run_dir=tmp_path / "ml-runs/run-1",
        run_id="run-1",
        pipeline_schema_version=1,
        pipeline_manifest_sha256="a" * 64,
        corpus_path=tmp_path / "ml-runs/run-1/07-corpus/final-corpus.jsonl",
        coordinates=np.zeros((4, 2), dtype=np.float32),
        labels=np.asarray([0, 0, 0, -1], dtype=np.int64),
        confidence=np.asarray([0.9, 0.8, 0.7, 0.0], dtype=np.float32),
        corpus=None,
        clustering=None,
        topics_manifest=None,
        topics=topics,
        reassignment=None,
        summary=AnalysisSummary(
            records=4,
            topics=1,
            outliers=1,
            outlier_share=0.25,
            mean_confidence=0.6,
            largest_topic_share=0.75,
            pipeline_status="awaiting_review",
            warnings=[],
        ),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _lineage_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    inspection_records: int = 6,
) -> AnalysisArtifacts:
    artifacts = _artifacts(tmp_path)
    run_dir = artifacts.run_dir
    inspection_path = run_dir / "01-inspection/dataset-profile.json"
    split_path = run_dir / "02-split/split-manifest.json"
    development_path = run_dir / "02-split/development.jsonl"
    cleaning_path = run_dir / "04-cleaning/cleaning-manifest.json"
    cleaned_path = run_dir / "04-cleaning/development-clean.jsonl"
    deduplication_path = run_dir / "06-deduplication/semantic-deduplication-manifest.json"
    for path in (inspection_path, split_path, cleaning_path, deduplication_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    development_path.write_text("parent-1\nparent-2\nparent-3\n", encoding="utf-8")
    cleaned_path.write_text("unit-1\nunit-2\nunit-3\nunit-4\n", encoding="utf-8")

    inspection = DatasetInspection.model_construct(contract_valid=inspection_records, sha256="a" * 64)
    split = DatasetSplitManifest.model_construct(
        source_sha256="a" * 64,
        stats=SplitStats(
            input_records=6,
            input_groups=6,
            assigned_records={SplitName.DEVELOPMENT: 3, SplitName.VALIDATION: 2, SplitName.TEST: 1},
            written_records={SplitName.DEVELOPMENT: 3, SplitName.VALIDATION: 2, SplitName.TEST: 1},
            group_counts={SplitName.DEVELOPMENT: 3, SplitName.VALIDATION: 2, SplitName.TEST: 1},
            removed_content_leaks={SplitName.DEVELOPMENT: 0, SplitName.VALIDATION: 0, SplitName.TEST: 0},
            ignored_noise_overlaps=0,
        ),
        output_sha256={
            SplitName.DEVELOPMENT: _sha256(development_path),
            SplitName.VALIDATION: "b" * 64,
            SplitName.TEST: "c" * 64,
        },
    )
    cleaning = DatasetCleaningManifest.model_construct(
        source_path=str(development_path),
        output_path=str(cleaned_path),
        output_sha256=_sha256(cleaned_path),
        stats=DatasetCleaningStats(
            input_rows=3,
            input_text_units=5,
            input_comments=3,
            input_replies=2,
            output_text_units=4,
            output_comments=3,
            output_replies=1,
            removed_by_reason={CleaningReason.TOO_SHORT: 1},
            detected_languages={"en": 5},
            duplicate_groups=0,
            largest_duplicate_group=0,
        ),
    )
    deduplication = SemanticDeduplicationManifest.model_construct(
        records_path=str(cleaned_path),
        result=DeduplicationStats(n_input=4, n_kept=4, n_removed=0, threshold=0.95),
    )
    corpus = CorpusManifest.model_construct(
        cleaning_manifest_sha256=_sha256(cleaning_path),
        deduplication_manifest_sha256=_sha256(deduplication_path),
        stats=CorpusStats(
            input_records=4,
            output_records=4,
            removed_semantic_duplicates=0,
            output_comments=3,
            output_replies=1,
            languages={"en": 4},
            unique_videos=2,
        ),
    )
    monkeypatch.setattr(reporting.DatasetInspection, "model_validate_json", lambda _value: inspection)
    monkeypatch.setattr(reporting.DatasetSplitManifest, "model_validate_json", lambda _value: split)
    monkeypatch.setattr(reporting.DatasetCleaningManifest, "model_validate_json", lambda _value: cleaning)
    monkeypatch.setattr(
        reporting.SemanticDeduplicationManifest,
        "model_validate_json",
        lambda _value: deduplication,
    )
    return AnalysisArtifacts(**{**artifacts.__dict__, "corpus": corpus})


def _card_artifacts(tmp_path: Path, *, reassigned: bool = False) -> AnalysisArtifacts:
    artifacts = _artifacts(tmp_path)
    summary_path = artifacts.run_dir / "09-clustering/cluster-summary.jsonl"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    cluster_summary = ClusterSummary(
        cluster_id=0,
        records=3,
        corpus_share=0.75,
        mean_probability=0.8,
        median_probability=0.82,
        minimum_probability=0.61,
        comments=2,
        replies=1,
        languages={"ru": 2, "en": 1},
        unique_videos=2,
        minimum_record_index=0,
    )
    summary_path.write_text(f"{cluster_summary.model_dump_json()}\n", encoding="utf-8")
    clustering = ClusteringManifest.model_construct(
        config=HDBSCANConfig(minimum_probability=0.5),
        summary_path=str(summary_path),
        summary_sha256=_sha256(summary_path),
    )
    topics_manifest = TopicRepresentationManifest.model_construct(
        representations_path=str(artifacts.run_dir / "10-topics/topic-representations.jsonl"),
    )
    result = replace(artifacts, clustering=clustering, topics_manifest=topics_manifest)
    if not reassigned:
        return result

    final_summary_path = artifacts.run_dir / "11-reassignment/final-cluster-summary.jsonl"
    final_summary_path.parent.mkdir(parents=True, exist_ok=True)
    final_summary = FinalClusterSummary(
        topic_id=0,
        original_records=3,
        reassigned_outliers=1,
        final_records=4,
        expansion_share=1 / 3,
        mean_reassignment_similarity=0.88,
        minimum_reassignment_similarity=0.88,
        comments=3,
        replies=1,
        languages={"ru": 3, "en": 1},
        unique_videos=3,
    )
    final_summary_path.write_text(f"{final_summary.model_dump_json()}\n", encoding="utf-8")
    reassignment = OutlierReassignmentManifest.model_construct(
        eligible_topics=[0],
        ineligible_topics={},
        summary_path=str(final_summary_path),
        summary_sha256=_sha256(final_summary_path),
    )
    final_analysis_summary = artifacts.summary.model_copy(
        update={"outliers": 0, "outlier_share": 0.0},
    )
    return replace(
        result,
        labels=np.asarray([0, 0, 0, 0], dtype=np.int64),
        confidence=np.asarray([0.9, 0.8, 0.7, 0.88], dtype=np.float32),
        reassignment=reassignment,
        summary=final_analysis_summary,
    )


def _excluded_card_artifacts(tmp_path: Path) -> AnalysisArtifacts:
    artifacts = _card_artifacts(tmp_path)
    final_summary_path = artifacts.run_dir / "11-reassignment/final-cluster-summary.jsonl"
    final_summary_path.parent.mkdir(parents=True, exist_ok=True)
    final_summary = FinalClusterSummary(
        topic_id=0,
        original_records=3,
        reassigned_outliers=0,
        final_records=3,
        expansion_share=0,
        comments=2,
        replies=1,
        languages={"ru": 2, "en": 1},
        unique_videos=2,
    )
    final_summary_path.write_text(f"{final_summary.model_dump_json()}\n", encoding="utf-8")
    reassignment = OutlierReassignmentManifest.model_construct(
        eligible_topics=[],
        ineligible_topics={0: "insufficient high-confidence centroid members"},
        summary_path=str(final_summary_path),
        summary_sha256=_sha256(final_summary_path),
    )
    return replace(artifacts, reassignment=reassignment)


def _representative_artifacts(tmp_path: Path) -> AnalysisArtifacts:
    artifacts = _card_artifacts(tmp_path)
    corpus_path = artifacts.run_dir / "07-corpus/final-corpus.jsonl"
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "corpus_id": f"corpus:{index:064x}",
            "cleaned_record_index": index,
            "record_id": f"record-{index}",
            "text": f"Private representative {index}",
            "clean_text": f"Private representative {index}",
            "text_kind": "reply" if index == 1 else "comment",
            "parent_record_id": "record-0" if index == 1 else None,
            "author": f"private-author-{index}",
            "detected_language": "ru",
            "video_id": "X3zn5uGvnaw" if index != 2 else "safeid2",
            "video_title": f"Video {index}",
            "video_channel": "Channel",
            "video_url": "https://www.youtube.com/watch?v=X3zn5uGvnaw" if index == 0 else "",
            "search_query": "delivery problems",
        }
        for index in range(4)
    ]
    corpus_path.write_text("".join(f"{json.dumps(row)}\n" for row in rows), encoding="utf-8")
    labels_path = artifacts.run_dir / "09-clustering/cluster-labels.npy"
    probabilities_path = artifacts.run_dir / "09-clustering/cluster-probabilities.npy"
    np.save(labels_path, np.asarray([0, 0, 0, -1], dtype=np.int64), allow_pickle=False)
    np.save(probabilities_path, np.asarray([0.91, 0.82, 0.73, 0.0], dtype=np.float32), allow_pickle=False)
    representative_path = artifacts.run_dir / "10-topics/representative-indices.jsonl"
    representative_path.parent.mkdir(parents=True, exist_ok=True)
    representatives = RepresentativeIndices(
        topic_id=0,
        record_indices=[0, 1, 2],
        centroid_similarities=[0.97, 0.92, 0.86],
    )
    representative_path.write_text(f"{representatives.model_dump_json()}\n", encoding="utf-8")
    topic = artifacts.topics[0].model_copy(update={"representative_indices": [0, 1, 2]})
    clustering = artifacts.clustering.model_copy(
        update={
            "labels_path": str(labels_path),
            "labels_sha256": _sha256(labels_path),
            "probabilities_path": str(probabilities_path),
            "probabilities_sha256": _sha256(probabilities_path),
        },
    )
    topics_manifest = artifacts.topics_manifest.model_copy(
        update={
            "representative_indices_path": str(representative_path),
            "representative_indices_sha256": _sha256(representative_path),
        },
    )
    return replace(
        artifacts,
        corpus_path=corpus_path,
        clustering=clustering,
        topics=(topic,),
        topics_manifest=topics_manifest,
    )


def _review_artifacts(tmp_path: Path) -> AnalysisArtifacts:
    artifacts = _representative_artifacts(tmp_path)
    existing = [json.loads(line) for line in artifacts.corpus_path.read_text(encoding="utf-8").splitlines()]
    for index in (4, 5):
        existing.append(
            {
                **existing[0],
                "corpus_id": f"corpus:{index:064x}",
                "cleaned_record_index": index,
                "record_id": f"record-{index}",
                "text": f"Private review {index}",
                "clean_text": f"Private review {index}",
                "author": f"private-author-{index}",
            },
        )
    artifacts.corpus_path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in existing),
        encoding="utf-8",
    )
    labels_path = Path(artifacts.clustering.labels_path)
    probabilities_path = Path(artifacts.clustering.probabilities_path)
    np.save(labels_path, np.asarray([0, 0, 0, -1, -1, -1], dtype=np.int64), allow_pickle=False)
    np.save(
        probabilities_path,
        np.asarray([0.91, 0.21, 0.73, 0.0, 0.0, 0.0], dtype=np.float32),
        allow_pickle=False,
    )
    clustering = artifacts.clustering.model_copy(
        update={"labels_sha256": _sha256(labels_path), "probabilities_sha256": _sha256(probabilities_path)},
    )
    decisions_path = artifacts.run_dir / "11-reassignment/outlier-decisions.jsonl"
    decisions_path.parent.mkdir(parents=True, exist_ok=True)
    decisions = (
        OutlierDecision(
            record_index=3,
            final_label=0,
            best_topic=0,
            best_similarity=0.87,
            second_topic=None,
            second_similarity=None,
            margin=0.06,
            reassigned=True,
            reason=OutlierDecisionReason.REASSIGNED,
        ),
        OutlierDecision(
            record_index=4,
            final_label=-1,
            best_topic=0,
            best_similarity=0.84,
            second_topic=None,
            second_similarity=None,
            margin=0.04,
            reassigned=False,
            reason=OutlierDecisionReason.BELOW_SIMILARITY,
        ),
        OutlierDecision(
            record_index=5,
            final_label=-1,
            best_topic=None,
            best_similarity=None,
            second_topic=None,
            second_similarity=None,
            margin=None,
            reassigned=False,
            reason=OutlierDecisionReason.NO_ELIGIBLE_TOPIC,
        ),
    )
    decisions_path.write_text(
        "".join(f"{decision.model_dump_json()}\n" for decision in decisions),
        encoding="utf-8",
    )
    reassignment = OutlierReassignmentManifest.model_construct(
        decisions_path=str(decisions_path),
        decisions_sha256=_sha256(decisions_path),
        config=OutlierReassignmentConfig(),
        metrics=OutlierReassignmentMetrics.model_construct(original_outliers=3),
    )
    return replace(
        artifacts,
        clustering=clustering,
        labels=np.asarray([0, 0, 0, 0, -1, -1], dtype=np.int64),
        confidence=np.asarray([0.91, 0.21, 0.73, 0.87, 0.84, 0.0], dtype=np.float32),
        reassignment=reassignment,
        summary=artifacts.summary.model_copy(update={"records": 6}),
    )


def _problem_artifacts(tmp_path: Path, texts: list[str] | None = None) -> AnalysisArtifacts:
    artifacts = _representative_artifacts(tmp_path)
    rows = [json.loads(line) for line in artifacts.corpus_path.read_text(encoding="utf-8").splitlines()]
    active_texts = texts or [
        "Есть проблема с доставкой",
        "Нет проблем, но есть задержка заказа",
        "Прекрасное видео",
        "Выброс с ошибкой",
    ]
    for index, (row, text) in enumerate(zip(rows, active_texts, strict=True)):
        row["text"] = text
        row["clean_text"] = text
        row["video_id"] = "video-a" if index < 2 else "video-b"
    artifacts.corpus_path.write_text(
        "".join(f"{json.dumps(row, ensure_ascii=False)}\n" for row in rows),
        encoding="utf-8",
    )
    corpus = CorpusManifest.model_construct(corpus_sha256=_sha256(artifacts.corpus_path))
    return replace(artifacts, corpus=corpus)


def test_stratified_plot_indices_are_deterministic_and_keep_every_label() -> None:
    labels = np.repeat(np.asarray([-1, 0, 1, 2], dtype=np.int64), [80, 10, 5, 5])

    first = stratified_plot_indices(labels, maximum=20, seed=7)
    second = stratified_plot_indices(labels, maximum=20, seed=7)

    np.testing.assert_array_equal(first, second)
    assert len(first) == 20
    assert set(labels[first]) == {-1, 0, 1, 2}


def test_topic_rows_and_reports_contain_only_aggregate_data(tmp_path: Path) -> None:
    artifacts = _artifacts(tmp_path)
    before = set(tmp_path.rglob("*"))

    rows = topic_summary_rows(artifacts)
    manifest = write_analysis_tables(artifacts, tmp_path / "visualizations")

    assert rows == [
        TopicSummaryRow(
            topic_id=0,
            name="delivery / order",
            records=3,
            corpus_share=0.75,
            mean_probability=0.8,
            unique_videos=2,
            keywords=["delivery"],
        ),
    ]
    assert isinstance(manifest, ReportManifest)
    assert not set(artifacts.run_dir.rglob("*")).difference(before)
    assert "comment_text" not in (tmp_path / "visualizations/run-1/summary.json").read_text()


def test_report_refuses_overwrite_and_output_inside_run(tmp_path: Path) -> None:
    artifacts = _artifacts(tmp_path)
    output = tmp_path / "visualizations"
    write_analysis_tables(artifacts, output)

    with pytest.raises(FileExistsError, match="already exists"):
        write_analysis_tables(artifacts, output)
    with pytest.raises(ValueError, match="outside"):
        write_analysis_tables(artifacts, artifacts.run_dir / "visualizations")


def test_data_lineage_explains_scopes_sources_and_flattening(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = _lineage_artifacts(tmp_path, monkeypatch)

    lineage = build_data_lineage(artifacts)

    assert lineage.metric("all_valid_parents").records == 6
    assert lineage.metric("all_valid_parents").scope == DataScope.ALL
    assert lineage.metric("development_parents").records == 3
    assert lineage.metric("development_replies").records == 2
    assert lineage.metric("development_flattened").records == 5
    assert lineage.metric("development_flattened").source.field == "stats.input_text_units"
    assert all(check.passed for check in lineage.checks)
    assert lineage.cleaning_removed_by_reason == {"too_short": 1}
    assert [step.records for step in processing_flow(artifacts)] == [3, 5, 4, 4, 4]


def test_data_lineage_blocks_mixed_or_inconsistent_run_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = _lineage_artifacts(tmp_path, monkeypatch, inspection_records=7)

    with pytest.raises(ValueError, match="inspection_to_split"):
        build_data_lineage(artifacts)


def test_data_lineage_blocks_tampered_development_split(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = _lineage_artifacts(tmp_path, monkeypatch)
    development_path = artifacts.run_dir / "02-split/development.jsonl"
    development_path.write_text("tampered\n", encoding="utf-8")

    with pytest.raises(ValueError, match="checksum mismatch"):
        build_data_lineage(artifacts)


def test_data_lineage_public_functions_have_docstrings() -> None:
    assert inspect.getdoc(build_data_lineage)
    assert inspect.getdoc(processing_flow)


def test_cluster_card_uses_verified_aggregate_hdbscan_summary(tmp_path: Path) -> None:
    artifacts = _card_artifacts(tmp_path)
    before = {path: path.stat().st_mtime_ns for path in artifacts.run_dir.rglob("*") if path.is_file()}

    cards = build_cluster_cards(artifacts)

    assert cards == [
        ClusterCard(
            topic_id=0,
            name="delivery / order",
            keywords=["delivery"],
            original_records=3,
            final_records=3,
            final_corpus_share=0.75,
            comments=2,
            replies=1,
            languages={"ru": 2, "en": 1},
            unique_videos=2,
            original_mean_probability=0.8,
            original_median_probability=0.82,
            original_minimum_probability=0.61,
            representative_records=1,
            reassigned_outliers=0,
            reassignment_status=ReassignmentStatus.NOT_RUN,
            pipeline_status="awaiting_review",
            warnings=[],
            sources=cards[0].sources,
        ),
    ]
    assert cards[0].sources["clustering"].field == "cluster_id=0"
    assert {path: path.stat().st_mtime_ns for path in artifacts.run_dir.rglob("*") if path.is_file()} == before
    assert get_cluster_card(cards, 0) == cards[0]
    with pytest.raises(KeyError, match="unknown topic ID"):
        get_cluster_card(cards, 99)


def test_cluster_card_keeps_reassignment_similarity_separate_from_probability(tmp_path: Path) -> None:
    card = build_cluster_cards(_card_artifacts(tmp_path, reassigned=True))[0]

    assert card.original_mean_probability == 0.8
    assert card.reassigned_outliers == 1
    assert card.mean_reassignment_similarity == 0.88
    assert card.final_records == 4
    assert card.reassignment_status == ReassignmentStatus.ELIGIBLE
    assert "reassignment" in card.sources


def test_cluster_cards_reject_tampered_summary(tmp_path: Path) -> None:
    artifacts = _card_artifacts(tmp_path)
    Path(artifacts.clustering.summary_path).write_text("tampered\n", encoding="utf-8")

    with pytest.raises(ValueError, match="checksum mismatch"):
        build_cluster_cards(artifacts)


def test_cluster_card_explains_reassignment_exclusion(tmp_path: Path) -> None:
    card = build_cluster_cards(_excluded_card_artifacts(tmp_path))[0]

    assert card.reassignment_status == ReassignmentStatus.EXCLUDED
    assert card.reassignment_exclusion_reason == "insufficient high-confidence centroid members"
    assert card.reassigned_outliers == 0
    assert "topic was excluded from outlier reassignment" in card.warnings


def test_representative_comments_use_dynamic_per_topic_limit_and_safe_sources(tmp_path: Path) -> None:
    artifacts = _representative_artifacts(tmp_path)
    before = {path: path.stat().st_mtime_ns for path in artifacts.run_dir.rglob("*") if path.is_file()}

    all_available = representative_comments(artifacts, n=None)
    limited = representative_comments(artifacts, n=2, topic_id=0)
    above_available = representative_comments(artifacts, n=100)

    assert len(all_available) == len(above_available) == 3
    assert [item.rank for item in limited] == [1, 2]
    assert [item.centroid_similarity for item in all_available] == [0.97, 0.92, 0.86]
    assert [item.hdbscan_probability for item in all_available] == pytest.approx([0.91, 0.82, 0.73])
    assert all_available[0].video_url == "https://www.youtube.com/watch?v=X3zn5uGvnaw"
    assert all_available[1].video_url == "https://www.youtube.com/watch?v=X3zn5uGvnaw"
    assert all_available[1].text_kind == "reply"
    assert "author" not in RepresentativeComment.model_fields
    assert {path: path.stat().st_mtime_ns for path in artifacts.run_dir.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("invalid_n", [0, -1, 1.5, True])
def test_representative_comments_reject_invalid_limit(tmp_path: Path, invalid_n: object) -> None:
    with pytest.raises(ValueError, match="positive integer or None"):
        representative_comments(_representative_artifacts(tmp_path), n=invalid_n)


def test_representative_comments_reject_tampered_indices(tmp_path: Path) -> None:
    artifacts = _representative_artifacts(tmp_path)
    Path(artifacts.topics_manifest.representative_indices_path).write_text("tampered\n", encoding="utf-8")

    with pytest.raises(ValueError, match="checksum mismatch"):
        representative_comments(artifacts)


def test_assignment_review_keeps_score_types_and_outlier_meanings_separate(tmp_path: Path) -> None:
    artifacts = _review_artifacts(tmp_path)
    before = {path: path.stat().st_mtime_ns for path in artifacts.run_dir.rglob("*") if path.is_file()}

    rows = assignment_review_comments(artifacts, n=2, topic_id=0)

    assert [row.review_kind for row in rows] == [
        AssignmentReviewKind.LOW_HDBSCAN_PROBABILITY,
        AssignmentReviewKind.LOW_HDBSCAN_PROBABILITY,
        AssignmentReviewKind.REASSIGNED_OUTLIER,
        AssignmentReviewKind.REJECTED_OUTLIER,
    ]
    assert [row.record_index for row in rows[:2]] == [1, 2]
    assert rows[0].hdbscan_probability == pytest.approx(0.21)
    assert rows[0].best_cosine_similarity is None
    assert rows[2].assigned_topic_id == rows[2].candidate_topic_id == 0
    assert rows[2].similarity_margin == 0.06
    assert rows[3].assigned_topic_id is None
    assert rows[3].candidate_topic_id == 0
    assert rows[3].decision_reason == OutlierDecisionReason.BELOW_SIMILARITY
    assert "author" not in AssignmentReviewComment.model_fields
    assert {path: path.stat().st_mtime_ns for path in artifacts.run_dir.rglob("*") if path.is_file()} == before


def test_assignment_review_works_without_reassignment_and_validates_limit(tmp_path: Path) -> None:
    artifacts = _representative_artifacts(tmp_path)

    rows = assignment_review_comments(artifacts, n=1)

    assert len(rows) == 1
    assert rows[0].record_index == 2
    assert rows[0].review_kind == AssignmentReviewKind.LOW_HDBSCAN_PROBABILITY
    for invalid_n in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="positive integer"):
            assignment_review_comments(artifacts, n=invalid_n)  # type: ignore[arg-type]


def test_assignment_review_includes_rejections_without_an_eligible_candidate(tmp_path: Path) -> None:
    rows = assignment_review_comments(_review_artifacts(tmp_path), n=2)

    no_candidate = next(row for row in rows if row.record_index == 5)
    assert no_candidate.review_kind == AssignmentReviewKind.REJECTED_OUTLIER
    assert no_candidate.assigned_topic_id is None
    assert no_candidate.candidate_topic_id is None
    assert no_candidate.topic_name is None
    assert no_candidate.decision_reason == OutlierDecisionReason.NO_ELIGIBLE_TOPIC


def test_assignment_review_rejects_tampered_outlier_decisions(tmp_path: Path) -> None:
    artifacts = _review_artifacts(tmp_path)
    Path(artifacts.reassignment.decisions_path).write_text("tampered\n", encoding="utf-8")

    with pytest.raises(ValueError, match="checksum mismatch"):
        assignment_review_comments(artifacts)


def test_explain_original_assignment_keeps_hdbscan_evidence_distinct(tmp_path: Path) -> None:
    explanation = explain_assignment(_review_artifacts(tmp_path), 1, representative_limit=2)

    assert isinstance(explanation, AssignmentExplanation)
    assert explanation.outcome == AssignmentOutcome.ORIGINAL_CLUSTER_MEMBER
    assert explanation.assigned_topic_id == 0
    assert explanation.candidate_topic_id is None
    assert explanation.hdbscan_probability == pytest.approx(0.21)
    assert explanation.best_cosine_similarity is None
    assert len(explanation.representative_context) == 2
    assert explanation.topic_keywords == ["delivery"]
    assert "author" not in AssignmentExplanation.model_fields


def test_explain_reassigned_outlier_reports_decision_and_thresholds(tmp_path: Path) -> None:
    explanation = explain_assignment(_review_artifacts(tmp_path), 3)

    assert explanation.outcome == AssignmentOutcome.REASSIGNED_OUTLIER
    assert explanation.assigned_topic_id == explanation.candidate_topic_id == 0
    assert explanation.hdbscan_probability is None
    assert explanation.best_cosine_similarity == 0.87
    assert explanation.similarity_margin == 0.06
    assert explanation.decision_reason == OutlierDecisionReason.REASSIGNED
    assert explanation.decision_thresholds == {
        "similarity_threshold": 0.85,
        "single_topic_similarity_threshold": 0.9,
        "margin_threshold": 0.05,
    }


def test_explain_remaining_outliers_distinguishes_candidate_from_assignment(tmp_path: Path) -> None:
    candidate = explain_assignment(_review_artifacts(tmp_path), 4, representative_limit=1)
    no_candidate = explain_assignment(_review_artifacts(tmp_path), 5)

    assert candidate.outcome == no_candidate.outcome == AssignmentOutcome.REMAINING_OUTLIER
    assert candidate.assigned_topic_id is None
    assert candidate.candidate_topic_id == 0
    assert candidate.decision_reason == OutlierDecisionReason.BELOW_SIMILARITY
    assert len(candidate.representative_context) == 1
    assert no_candidate.assigned_topic_id is no_candidate.candidate_topic_id is None
    assert no_candidate.representative_context == []
    assert no_candidate.decision_reason == OutlierDecisionReason.NO_ELIGIBLE_TOPIC


def test_explain_assignment_validates_indices_limits_and_final_alignment(tmp_path: Path) -> None:
    artifacts = _review_artifacts(tmp_path)
    for invalid_index in (-1, 1.5, True):
        with pytest.raises(ValueError, match="non-negative integer"):
            explain_assignment(artifacts, invalid_index)  # type: ignore[arg-type]
    with pytest.raises(IndexError, match="outside the corpus"):
        explain_assignment(artifacts, 6)
    with pytest.raises(ValueError, match="non-negative integer"):
        explain_assignment(artifacts, 0, representative_limit=-1)
    inconsistent = replace(artifacts, labels=np.asarray([0, 0, 0, -1, -1, -1], dtype=np.int64))
    with pytest.raises(ValueError, match="disagrees with the final labels"):
        explain_assignment(inconsistent, 3)


def test_problem_signal_assessment_is_aggregate_multilingual_triage(tmp_path: Path) -> None:
    artifacts = _problem_artifacts(tmp_path)
    config = ProblemSignalConfig(
        minimum_signal_records=2,
        problem_candidate_minimum_share=0.5,
        topic_only_maximum_share=0.1,
    )

    assessment = assess_topic_problem_signals(artifacts, config=config)[0]

    assert assessment.status == ProblemAssessmentStatus.PROBLEM_CANDIDATE
    assert assessment.records == 3
    assert assessment.signal_records == 2
    assert assessment.signal_share == 2 / 3
    assert assessment.signal_comments == 1
    assert assessment.signal_replies == 1
    assert assessment.unique_videos == 2
    assert assessment.signal_videos == 1
    assert assessment.signal_video_share == 0.5
    assert assessment.signals == {"задерж": 1, "проблем": 1}
    assert assessment.requires_manual_review is True
    assert "Private" not in assessment.model_dump_json()


def test_problem_signal_negations_and_thresholds_avoid_false_certainty(tmp_path: Path) -> None:
    negated = _problem_artifacts(
        tmp_path / "negated",
        ["Нет проблем", "Всё без проблем", "Обычная тема", "problem outlier"],
    )
    uncertain = _problem_artifacts(
        tmp_path / "uncertain",
        ["Ошибка оплаты", "Обычная тема", "Обычная тема", "problem outlier"],
    )
    config = ProblemSignalConfig(
        minimum_signal_records=2,
        problem_candidate_minimum_share=0.5,
        topic_only_maximum_share=0.1,
    )

    topic_only = assess_topic_problem_signals(negated, config=config)[0]
    uncertain_result = assess_topic_problem_signals(uncertain, config=config)[0]

    assert topic_only.status == ProblemAssessmentStatus.TOPIC_ONLY
    assert topic_only.signal_records == 0
    assert topic_only.requires_manual_review is False
    assert uncertain_result.status == ProblemAssessmentStatus.UNCERTAIN
    assert uncertain_result.signal_records == 1
    assert uncertain_result.requires_manual_review is True


def test_problem_signal_report_is_checksum_bound_aggregate_and_outside_run(tmp_path: Path) -> None:
    artifacts = _problem_artifacts(tmp_path)
    output = tmp_path / "visualizations" / artifacts.run_id
    config = ProblemSignalConfig(minimum_signal_records=1)

    manifest = write_problem_signal_report(artifacts, output, config=config)

    report_text = (output / "problem-signals.jsonl").read_text(encoding="utf-8")
    assert manifest.topics == 1
    assert manifest.private_text_included is False
    assert manifest.corpus_sha256 == artifacts.corpus.corpus_sha256
    assert manifest.statuses[ProblemAssessmentStatus.PROBLEM_CANDIDATE] == 1
    assert "Есть проблема" not in report_text
    assert "private-author" not in report_text
    with pytest.raises(FileExistsError, match="already exists"):
        write_problem_signal_report(artifacts, output, config=config)
    with pytest.raises(ValueError, match="outside"):
        write_problem_signal_report(artifacts, artifacts.run_dir / "report", config=config)


def test_problem_signal_assessment_rejects_tampered_corpus_and_invalid_policy(tmp_path: Path) -> None:
    artifacts = _problem_artifacts(tmp_path)
    artifacts.corpus_path.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        assess_topic_problem_signals(artifacts)
    with pytest.raises(ValueError, match="must exceed"):
        ProblemSignalConfig(
            problem_candidate_minimum_share=0.1,
            topic_only_maximum_share=0.1,
        )


def test_problem_priority_is_relative_explainable_and_stably_ranked(tmp_path: Path) -> None:
    assessments = assess_topic_problem_signals(
        _problem_artifacts(tmp_path),
        config=ProblemSignalConfig(
            minimum_signal_records=2,
            problem_candidate_minimum_share=0.5,
            topic_only_maximum_share=0.1,
        ),
    )
    base = assessments[0]
    lower = base.model_copy(
        update={
            "topic_id": 1,
            "topic_name": "Smaller signal",
            "signal_records": 1,
            "signal_share": 1 / 3,
            "status": ProblemAssessmentStatus.UNCERTAIN,
        },
    )
    topic_only = base.model_copy(
        update={
            "topic_id": 2,
            "topic_name": "Ordinary topic",
            "signal_records": 0,
            "signal_share": 0.0,
            "signal_videos": 0,
            "signal_video_share": 0.0,
            "status": ProblemAssessmentStatus.TOPIC_ONLY,
        },
    )

    priorities = rank_problem_topics([lower, topic_only, base])

    assert [priority.topic_id for priority in priorities] == [0, 1]
    assert [priority.rank for priority in priorities] == [1, 2]
    assert priorities[0].priority_score > priorities[1].priority_score
    assert priorities[0].signal_volume_component == 1
    assert priorities[0].topic_scale_component == 1
    assert "not verified severity" in priorities[0].interpretation


def test_problem_priority_policy_and_aggregate_validation(tmp_path: Path) -> None:
    assessment = assess_topic_problem_signals(_problem_artifacts(tmp_path))[0]

    assert rank_problem_topics(
        [assessment],
        config=ProblemPriorityConfig(include_uncertain=False),
    ) == []
    with pytest.raises(ValueError, match=r"sum to 1\.0"):
        ProblemPriorityConfig(signal_share_weight=0.5)
    with pytest.raises(ValueError, match="duplicate topic IDs"):
        rank_problem_topics([assessment, assessment])
    invalid = assessment.model_copy(update={"signal_records": assessment.records + 1})
    with pytest.raises(ValueError, match="more signal rows"):
        rank_problem_topics([invalid])


def test_problem_priority_report_is_checksum_bound_private_free_and_outside_run(tmp_path: Path) -> None:
    artifacts = _problem_artifacts(tmp_path)
    output = tmp_path / "visualizations" / artifacts.run_id

    manifest = write_problem_priority_report(artifacts, output)

    report_text = (output / "problem-priorities.jsonl").read_text(encoding="utf-8")
    assert manifest.ranked_topics == 1
    assert manifest.private_text_included is False
    assert len(manifest.assessments_sha256) == 64
    assert manifest.signal_config == ProblemSignalConfig()
    assert "Есть проблема" not in report_text
    assert "private-author" not in report_text
    with pytest.raises(FileExistsError, match="already exists"):
        write_problem_priority_report(artifacts, output)
    with pytest.raises(ValueError, match="outside"):
        write_problem_priority_report(artifacts, artifacts.run_dir / "report")
