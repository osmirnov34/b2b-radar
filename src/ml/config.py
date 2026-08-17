from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.ml.schemas import ANALYSIS_SCHEMA_VERSION


class _ConfigModel(BaseModel):
    """Strict immutable base for reproducible analysis configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class CleaningConfig(_ConfigModel):
    schema_version: int = 1
    min_length: int = Field(default=20, ge=0, le=10_000)
    max_length: int | None = Field(default=None, ge=1, le=1_000_000)
    unicode_normalization: Literal["NFC", "NFKC"] = "NFKC"
    strip_html: bool = True
    url_handling: Literal["keep", "token", "remove"] = "token"
    acknowledgement_filter: bool = True
    spam_filter: bool = False
    repeated_char_filter: bool = True
    uppercase_filter: bool = False
    detect_language: bool = True
    allowed_languages: tuple[str, ...] = ()
    exact_deduplication: bool = True

    @field_validator("allowed_languages")
    @classmethod
    def normalize_languages(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(value.strip().casefold() for value in values if value.strip()))

    @model_validator(mode="after")
    def validate_lengths(self) -> "CleaningConfig":
        if self.max_length is not None and self.max_length < self.min_length:
            msg = "max_length cannot be less than min_length"
            raise ValueError(msg)
        return self


class EmbeddingConfig(_ConfigModel):
    model_name: str = Field(default="intfloat/multilingual-e5-large", min_length=1)
    threads: int = Field(default=4, ge=1)
    batch_size: int = Field(default=64, ge=1)
    max_seq_length: int = Field(default=512, ge=8)

    @field_validator("model_name")
    @classmethod
    def normalize_model_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            msg = "model_name cannot be blank"
            raise ValueError(msg)
        return normalized


class DeduplicationConfig(_ConfigModel):
    enabled: bool = True
    threshold: float = Field(default=0.95, ge=0, le=1)
    block_size: int = Field(default=2048, ge=1)
    sample_pairs: int = Field(default=8, ge=0)
    backend: Literal["hnsw", "exhaustive"] = "hnsw"
    exhaustive_max_records: int = Field(default=50_000, ge=2)
    ann_neighbors: int = Field(default=64, ge=2)
    ann_ef_construction: int = Field(default=200, ge=2)
    ann_ef_search: int = Field(default=200, ge=2)
    ann_m: int = Field(default=16, ge=2)
    random_seed: int = 42
    separate_text_kinds: bool = True


class ClusteringConfig(_ConfigModel):
    min_topic_size: int = Field(default=250, ge=2)
    reduce_outliers: bool = True
    reduce_outliers_threshold: float = Field(default=0.9, ge=0, le=1)
    random_seed: int = 42
    top_n: int = Field(default=50, ge=0)


class AnalysisConfig(_ConfigModel):
    schema_version: int = ANALYSIS_SCHEMA_VERSION
    input_path: Path
    output_dir: Path = Path("docs/analysis-output")
    limit: int | None = Field(default=None, ge=1)
    sample_size: int | None = Field(default=None, ge=1)
    cleaning: CleaningConfig = Field(default_factory=CleaningConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    deduplication: DeduplicationConfig = Field(default_factory=DeduplicationConfig)
    clustering: ClusteringConfig = Field(default_factory=ClusteringConfig)

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, value: int) -> int:
        if value != ANALYSIS_SCHEMA_VERSION:
            msg = f"unsupported analysis config schema_version={value}; expected {ANALYSIS_SCHEMA_VERSION}"
            raise ValueError(msg)
        return value

    @classmethod
    def load_json(cls, path: Path) -> Self:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{self.model_dump_json(indent=2)}\n", encoding="utf-8")


def load_analysis_config(path: Path) -> AnalysisConfig:
    return AnalysisConfig.load_json(path)


def save_analysis_config(config: AnalysisConfig, path: Path) -> None:
    config.save_json(path)
