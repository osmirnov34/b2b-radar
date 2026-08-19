from pathlib import Path

import pytest
from pydantic import ValidationError

from src.ml import CleaningConfig, DeduplicationConfig, EmbeddingConfig

SEMANTIC_DEDUP_CONFIG = Path(__file__).parents[1] / "configs" / "semantic-deduplication.example.json"
DATASET_CLEANING_CONFIG = Path(__file__).parents[1] / "configs" / "dataset-cleaning.example.json"
EMBEDDINGS_CONFIG = Path(__file__).parents[1] / "configs" / "embeddings.example.json"


def test_stage_config_models_are_immutable_and_forbid_extra_fields() -> None:
    config = EmbeddingConfig()

    with pytest.raises(ValidationError, match="frozen"):
        config.batch_size = 10
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        EmbeddingConfig.model_validate({"unknown": True})


@pytest.mark.parametrize("threshold", [-0.01, 1.01])
def test_deduplication_threshold_must_be_probability(threshold: float) -> None:
    with pytest.raises(ValidationError):
        DeduplicationConfig(threshold=threshold)


def test_deduplication_ann_defaults_are_production_safe() -> None:
    config = DeduplicationConfig()

    assert config.backend == "hnsw"
    assert config.ann_neighbors == 64
    assert config.separate_text_kinds is True


def test_stage_configs_reject_invalid_operational_ranges() -> None:
    with pytest.raises(ValidationError):
        EmbeddingConfig(threads=0)
    with pytest.raises(ValidationError):
        DeduplicationConfig(block_size=0)


def test_embedding_model_name_is_trimmed_and_cannot_be_blank() -> None:
    assert EmbeddingConfig(model_name="  model/name  ").model_name == "model/name"
    with pytest.raises(ValidationError, match="model_name cannot be blank"):
        EmbeddingConfig(model_name="   ")


def test_stage_specific_example_configs_match_contracts() -> None:
    cleaning = CleaningConfig.model_validate_json(DATASET_CLEANING_CONFIG.read_text(encoding="utf-8"))
    deduplication = DeduplicationConfig.model_validate_json(SEMANTIC_DEDUP_CONFIG.read_text(encoding="utf-8"))
    embedding = EmbeddingConfig.model_validate_json(EMBEDDINGS_CONFIG.read_text(encoding="utf-8"))

    assert cleaning.url_handling == "token"
    assert deduplication.backend == "hnsw"
    assert embedding.model_name == "intfloat/multilingual-e5-large"
