import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.analysis.schemas import AnalysisRunMetadata, ClusterRecord, ExportedComment
from src.web.routers.analysis import _parse_clusters

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _jsonl_lines(name: str) -> list[str]:
    return [line for line in (FIXTURES_DIR / name).read_text(encoding="utf-8").splitlines() if line.strip()]


def test_comments_export_fixture_matches_input_contract() -> None:
    comments = [ExportedComment.model_validate_json(line) for line in _jsonl_lines("comments.jsonl")]

    assert len(comments) == 2
    assert comments[0].comment_id == "comment-1"
    assert comments[0].comment_published_at is not None


def test_comments_contract_normalizes_nullable_provenance() -> None:
    comment = ExportedComment.model_validate(
        {"comment_text": "Text", "comment_author": None, "video_channel": None, "search_query": None},
    )

    assert comment.comment_author == ""
    assert comment.video_channel == ""
    assert comment.search_query == ""


def test_clusters_fixture_matches_contract_and_web_import() -> None:
    raw = (FIXTURES_DIR / "clusters.jsonl").read_bytes()
    records = [ClusterRecord.model_validate_json(line) for line in _jsonl_lines("clusters.jsonl")]
    web_records = _parse_clusters(raw)

    assert len(records) == 1
    assert records[0].n_comments == len(records[0].comments)
    assert web_records == [json.loads(line) for line in _jsonl_lines("clusters.jsonl")]


def test_run_metadata_fixture_matches_contract() -> None:
    metadata = AnalysisRunMetadata.model_validate_json((FIXTURES_DIR / "run_meta.json").read_text(encoding="utf-8"))

    assert metadata.schema_version == 1
    assert metadata.n_after_dedup <= metadata.n_after_clean <= metadata.n_input


def test_legacy_run_metadata_defaults_to_schema_version_one() -> None:
    data = json.loads((FIXTURES_DIR / "run_meta.json").read_text(encoding="utf-8"))
    del data["schema_version"]

    assert AnalysisRunMetadata.model_validate(data).schema_version == 1


def test_cluster_contract_rejects_mismatched_comment_count() -> None:
    with pytest.raises(ValidationError, match="does not match comments length"):
        ClusterRecord.model_validate(
            {
                "topic_id": 0,
                "n_comments": 2,
                "n_authors": 1,
                "n_channels": 1,
                "comments": [{"text": "Only one comment"}],
            },
        )


def test_run_metadata_contract_rejects_impossible_processing_counts() -> None:
    data = json.loads((FIXTURES_DIR / "run_meta.json").read_text(encoding="utf-8"))
    data["n_after_dedup"] = data["n_after_clean"] + 1

    with pytest.raises(ValidationError, match="n_after_dedup cannot exceed n_after_clean"):
        AnalysisRunMetadata.model_validate(data)
