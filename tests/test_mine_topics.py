from pathlib import Path

import pytest

from scripts.mine_topics import build_parser, clean, clustering_prefix, config_from_args, parse_config


def test_model_specific_clustering_prefixes() -> None:
    assert clustering_prefix("intfloat/multilingual-e5-large") == "query: "
    assert clustering_prefix("ai-forever/FRIDA") == "categorize_topic: "
    assert clustering_prefix("deepvk/USER-bge-m3") == ""


def test_clean_with_spam_filter_uses_pipeline_noise_gate() -> None:
    rows = [
        {"text": "Нам нужен простой CRM для работы с клиентами"},
        {"text": "оченьдлинноесклеенноесловобезпробелов"},
    ]

    assert clean(rows, min_length=10, spam_filter=True) == [rows[0]]


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
