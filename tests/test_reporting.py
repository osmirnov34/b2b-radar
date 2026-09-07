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
from src.ml.outlier_reassignment import FinalClusterSummary, OutlierReassignmentManifest
from src.ml.reporting import (
    AnalysisArtifacts,
    AnalysisSummary,
    ClusterCard,
    DataScope,
    ReassignmentStatus,
    ReportManifest,
    RepresentativeComment,
    TopicSummaryRow,
    build_cluster_cards,
    build_data_lineage,
    get_cluster_card,
    processing_flow,
    representative_comments,
    stratified_plot_indices,
    topic_summary_rows,
    write_analysis_tables,
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
