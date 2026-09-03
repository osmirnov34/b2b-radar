from pathlib import Path

import numpy as np
import pytest

from src.ml.reporting import (
    AnalysisArtifacts,
    AnalysisSummary,
    ReportManifest,
    TopicSummaryRow,
    stratified_plot_indices,
    topic_summary_rows,
    write_analysis_tables,
)
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
