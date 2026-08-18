from pathlib import Path

import pytest

from src.ml.export import (
    AssignmentSource,
    ExportConfig,
    ExportedAssignment,
    ExportManifest,
    ResearchAssignment,
    _safe_keywords,
    _safe_topic_name,
    export_topic_results,
)
from src.ml.topic_representation import TopicKeyword, TopicRepresentation
from tests.test_evaluation import _run_evaluation


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    _, evaluation_dir = _run_evaluation(tmp_path)
    return (
        tmp_path / "corpus-manifest.json",
        tmp_path / "clustering-manifest.json",
        tmp_path / "topics" / "topic-representation-manifest.json",
        tmp_path / "reassignment" / "outlier-reassignment-manifest.json",
        evaluation_dir / "evaluation-manifest.json",
    )


def test_public_export_is_complete_safe_and_auditable(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    output = tmp_path / "export"

    manifest = export_topic_results(*inputs, output)

    assignments = [
        ExportedAssignment.model_validate_json(line)
        for line in (output / "assignments.jsonl").read_text().splitlines()
    ]
    combined = "\n".join(
        (output / name).read_text()
        for name in ("topics.jsonl", "assignments.jsonl", "quality.json", "export-manifest.json")
    )
    assert len(assignments) == 7
    assert {item.source for item in assignments} == {AssignmentSource.HDBSCAN, AssignmentSource.REASSIGNMENT}
    assert all(item.corpus_id.startswith("corpus:") for item in assignments)
    assert "Private outlier" not in combined
    assert "author-" not in combined
    assert "video-" not in combined
    assert not (output / "research-assignments.jsonl").exists()
    assert ExportManifest.model_validate_json((output / "export-manifest.json").read_text()) == manifest
    assert manifest.records == 7
    assert manifest.outliers == 0


def test_sensitive_research_export_requires_explicit_configuration(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    output = tmp_path / "research-export"

    manifest = export_topic_results(*inputs, output, config=ExportConfig(include_research_text=True))

    records = [
        ResearchAssignment.model_validate_json(line)
        for line in (output / "research-assignments.jsonl").read_text().splitlines()
    ]
    assert len(records) == manifest.records
    assert any("Private outlier" in item.text for item in records)
    assert manifest.research_assignments_sha256 is not None


def test_export_rejects_preliminary_evaluation_and_protects_outputs(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    with pytest.raises(ValueError, match="completed manual review"):
        export_topic_results(
            *inputs,
            tmp_path / "final-only",
            config=ExportConfig(require_final_evaluation=True),
        )

    output = tmp_path / "protected"
    export_topic_results(*inputs, output)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        export_topic_results(*inputs, output)
    export_topic_results(*inputs, output, force=True)


def test_public_export_is_deterministic_and_rejects_tampered_metrics(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    export_topic_results(*inputs, first)
    export_topic_results(*inputs, second)
    for name in ("topics.jsonl", "assignments.jsonl", "quality.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()

    metrics = tmp_path / "evaluation" / "evaluation-metrics.json"
    metrics.write_text(metrics.read_text() + "\n")
    with pytest.raises(ValueError, match="evaluation metrics"):
        export_topic_results(*inputs, tmp_path / "tampered")


def test_public_topic_fields_filter_sensitive_terms() -> None:
    topic = TopicRepresentation(
        topic_id=3,
        name="sales@example.com / crm",
        records=2,
        mean_probability=0.8,
        languages={"en": 2},
        unique_videos=1,
        keywords=[
            TopicKeyword(term="sales@example.com", weight=1.0, rank=1, kind="unigram"),
            TopicKeyword(term="crm", weight=0.5, rank=2, kind="unigram"),
        ],
        representative_indices=[0],
    )
    keywords = _safe_keywords(topic, ExportConfig())
    assert keywords == ["crm"]
    assert _safe_topic_name(topic, keywords) == "crm"
