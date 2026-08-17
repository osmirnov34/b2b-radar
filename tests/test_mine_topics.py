from pathlib import Path

import pytest

from scripts.mine_topics import (
    build_parser,
    clustering_prefix,
    config_from_args,
    load_comments,
    parse_config,
    write_results,
)
from src.ml import ClusterRecord, CommentRecord

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_model_specific_clustering_prefixes() -> None:
    assert clustering_prefix("intfloat/multilingual-e5-large") == "query: "
    assert clustering_prefix("ai-forever/FRIDA") == "categorize_topic: "
    assert clustering_prefix("deepvk/USER-bge-m3") == ""


def test_load_comments_returns_normalized_typed_records() -> None:
    comments = load_comments(FIXTURES_DIR / "comments.jsonl")

    assert len(comments) == 2
    assert all(isinstance(comment, CommentRecord) for comment in comments)
    assert comments[0].author == "Анна"
    assert comments[0].video_url == "https://www.youtube.com/watch?v=video-1"


class _TopicColumn:
    def __ge__(self, _value: int) -> "_TopicColumn":
        return self

    @staticmethod
    def sum() -> int:
        return 1


class _TopicInfo:
    def __getitem__(self, _key: str) -> _TopicColumn:
        return _TopicColumn()

    @staticmethod
    def iterrows() -> list[tuple[int, dict[str, int]]]:
        return [(0, {"Topic": 0, "Count": 2})]


class _TopicModel:
    @staticmethod
    def get_topic_info() -> _TopicInfo:
        return _TopicInfo()

    @staticmethod
    def get_topic(_topic_id: int) -> list[tuple[str, float]]:
        return [("crm", 0.8), ("клиенты", 0.7)]


def test_write_results_preserves_cluster_contract(tmp_path: Path) -> None:
    comments = [
        CommentRecord(text="Первый комментарий", author="Анна", channel="Канал 1", video_id="v1"),
        CommentRecord(text="Второй комментарий", author="Иван", channel="Канал 2", video_id="v2"),
    ]

    n_topics, n_outliers = write_results(_TopicModel(), comments, [0, 0], tmp_path, top_n=1)
    record = ClusterRecord.model_validate_json((tmp_path / "clusters.jsonl").read_text(encoding="utf-8"))

    assert (n_topics, n_outliers) == (1, 0)
    assert record.n_comments == 2
    assert record.comments[0].video_url == "https://www.youtube.com/watch?v=v1"


def test_cli_defaults_map_to_internal_config() -> None:
    config = config_from_args(build_parser().parse_args(["comments.jsonl"]))

    assert config.input_path == Path("comments.jsonl")
    assert config.limit is None
    assert config.sample_size is None
    assert config.embedding.batch_size == 64
    assert config.deduplication.enabled is True
    assert config.clustering.reduce_outliers is True


def test_legacy_cli_flags_map_to_nested_config() -> None:
    args = build_parser().parse_args(
        [
            "comments.jsonl",
            "--model",
            "deepvk/USER-bge-m3",
            "--threads",
            "8",
            "--batch-size",
            "16",
            "--limit",
            "100",
            "--sample",
            "200",
            "--near-dup-threshold",
            "0.92",
            "--no-near-dup",
            "--no-reduce-outliers",
        ],
    )

    config = config_from_args(args)

    assert config.embedding.model_name == "deepvk/USER-bge-m3"
    assert config.embedding.threads == 8
    assert config.embedding.batch_size == 16
    assert config.limit == 100
    assert config.sample_size == 200
    assert config.deduplication.threshold == 0.92
    assert config.deduplication.enabled is False
    assert config.clustering.reduce_outliers is False


def test_cli_reports_invalid_config_without_traceback(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        parse_config(["comments.jsonl", "--threads", "0"])

    assert exc_info.value.code == 2
    assert "invalid analysis configuration" in capsys.readouterr().err
