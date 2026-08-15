from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ANALYSIS_SCHEMA_VERSION = 1


class _ContractModel(BaseModel):
    """Base policy for persisted analysis formats.

    Extra fields remain allowed for forward-compatible readers: producers may add metadata without
    making an older web application reject an otherwise compatible record.
    """

    model_config = ConfigDict(extra="ignore")


class ExportedComment(_ContractModel):
    """One line from the web application's comments JSONL export."""

    comment_text: str
    comment_id: str = ""
    comment_author: str = ""
    comment_published_at: datetime | None = None
    comment_like_count: int = Field(default=0, ge=0)
    comment_total_reply_count: int = Field(default=0, ge=0)
    video_id: str = ""
    video_title: str = ""
    video_channel: str = ""
    video_url: str = ""
    search_query: str = ""

    @field_validator(
        "comment_id",
        "comment_author",
        "video_id",
        "video_title",
        "video_channel",
        "video_url",
        "search_query",
        mode="before",
    )
    @classmethod
    def normalize_nullable_provenance(cls, value: object) -> object:
        return "" if value is None else value


class ClusterComment(_ContractModel):
    """Comment provenance embedded in a cluster record."""

    text: str
    author: str = ""
    channel: str = ""
    query: str = ""
    video_id: str = ""
    video_title: str = ""
    video_url: str = ""


class ClusterRecord(_ContractModel):
    """One non-outlier topic written as a line of clusters.jsonl."""

    topic_id: int = Field(ge=0)
    n_comments: int = Field(ge=0)
    n_authors: int = Field(ge=0)
    n_channels: int = Field(ge=0)
    keywords: list[str] = Field(default_factory=list)
    comments: list[ClusterComment] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_comment_count(self) -> "ClusterRecord":
        if self.n_comments != len(self.comments):
            msg = f"n_comments={self.n_comments} does not match comments length={len(self.comments)}"
            raise ValueError(msg)
        return self


class AnalysisRunMetadata(_ContractModel):
    """Reproducibility metadata persisted alongside clusters.jsonl."""

    schema_version: int = Field(default=ANALYSIS_SCHEMA_VERSION, ge=1)
    model: str = Field(min_length=1)
    near_dup_threshold: float | None = Field(default=None, ge=0, le=1)
    reduce_outliers_threshold: float | None = Field(default=None, ge=0, le=1)
    min_topic_size: int = Field(ge=2)
    n_input: int = Field(ge=0)
    n_after_clean: int = Field(ge=0)
    n_after_dedup: int = Field(ge=0)
    n_topics: int = Field(ge=0)
    n_outliers_before_reduction: int = Field(ge=0)
    n_outliers: int = Field(ge=0)
    created_at: datetime

    @model_validator(mode="after")
    def validate_processing_counts(self) -> "AnalysisRunMetadata":
        if self.n_after_clean > self.n_input:
            msg = "n_after_clean cannot exceed n_input"
            raise ValueError(msg)
        if self.n_after_dedup > self.n_after_clean:
            msg = "n_after_dedup cannot exceed n_after_clean"
            raise ValueError(msg)
        if self.n_outliers > self.n_outliers_before_reduction:
            msg = "n_outliers cannot exceed n_outliers_before_reduction"
            raise ValueError(msg)
        return self
