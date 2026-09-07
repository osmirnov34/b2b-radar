import hashlib
import inspect
from pathlib import Path

import numpy as np
import pytest

from src.ml import reporting
from src.ml.cleaning_dataset import DatasetCleaningManifest, DatasetCleaningStats
from src.ml.corpus import CorpusManifest, CorpusStats
from src.ml.inspection import DatasetInspection
from src.ml.models import CleaningReason, DeduplicationStats
from src.ml.reporting import (
    AnalysisArtifacts,
    AnalysisSummary,
    DataScope,
    ReportManifest,
    TopicSummaryRow,
    build_data_lineage,
    processing_flow,
    stratified_plot_indices,
    topic_summary_rows,
    write_analysis_tables,
)
from src.ml.semantic_deduplication import SemanticDeduplicationManifest
from src.ml.splitting import DatasetSplitManifest, SplitName, SplitStats
from src.ml.topic_representation import TopicKeyword, TopicRepresentation


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
