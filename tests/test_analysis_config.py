from pathlib import Path

import pytest
from pydantic import ValidationError

from src.analysis import (
    AnalysisConfig,
    ClusteringConfig,
    DeduplicationConfig,
    EmbeddingConfig,
    load_analysis_config,
    save_analysis_config,
)

EXAMPLE_CONFIG = Path(__file__).parents[1] / "configs" / "topic-analysis.example.json"


def test_analysis_config_defaults() -> None:
    config = AnalysisConfig(input_path=Path("comments.jsonl"))

    assert config.embedding.model_name == "intfloat/multilingual-e5-large"
    assert config.embedding.batch_size == 64
    assert config.deduplication.enabled is True
    assert config.clustering.reduce_outliers is True
    assert config.limit is None
    assert config.sample_size is None


def test_config_models_are_immutable_and_forbid_extra_fields() -> None:
    config = AnalysisConfig(input_path=Path("comments.jsonl"))

    with pytest.raises(ValidationError, match="frozen"):
        config.limit = 10
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        AnalysisConfig.model_validate({"input_path": "comments.jsonl", "unknown": True})


@pytest.mark.parametrize("threshold", [-0.01, 1.01])
def test_deduplication_threshold_must_be_probability(threshold: float) -> None:
    with pytest.raises(ValidationError):
        DeduplicationConfig(threshold=threshold)


def test_config_rejects_invalid_operational_ranges() -> None:
    with pytest.raises(ValidationError):
        EmbeddingConfig(threads=0)
    with pytest.raises(ValidationError):
        DeduplicationConfig(block_size=0)
    with pytest.raises(ValidationError):
        ClusteringConfig(min_topic_size=1)
    with pytest.raises(ValidationError):
        AnalysisConfig(input_path=Path("comments.jsonl"), limit=0)
    with pytest.raises(ValidationError):
        AnalysisConfig(input_path=Path("comments.jsonl"), sample_size=0)


def test_embedding_model_name_is_trimmed_and_cannot_be_blank() -> None:
    assert EmbeddingConfig(model_name="  model/name  ").model_name == "model/name"
    with pytest.raises(ValidationError, match="model_name cannot be blank"):
        EmbeddingConfig(model_name="   ")


def test_example_config_matches_contract() -> None:
    config = load_analysis_config(EXAMPLE_CONFIG)

    assert config.schema_version == 1
    assert config.input_path == Path("data/comments.jsonl")


def test_config_json_round_trip(tmp_path: Path) -> None:
    source = AnalysisConfig(
        input_path=Path("comments.jsonl"),
        limit=100,
        embedding=EmbeddingConfig(model_name="deepvk/USER-bge-m3", batch_size=16),
    )
    path = tmp_path / "nested" / "analysis.json"

    save_analysis_config(source, path)

    assert load_analysis_config(path) == source


def test_config_rejects_unsupported_schema_version() -> None:
    with pytest.raises(ValidationError, match="unsupported analysis config schema_version"):
        AnalysisConfig(input_path=Path("comments.jsonl"), schema_version=2)
