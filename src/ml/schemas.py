import hashlib
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ANALYSIS_SCHEMA_VERSION = 1


class _ContractModel(BaseModel):
    """Base policy for persisted analysis formats.

    Extra fields remain allowed for forward-compatible readers: producers may add metadata without
    making an older web application reject an otherwise compatible record.
    """

    model_config = ConfigDict(extra="ignore")


class TextKind(StrEnum):
    COMMENT = "comment"
    REPLY = "reply"


class ExportedReply(_ContractModel):
    """A nested reply; its metadata may be absent, but its text ownership is explicit."""

    text: str
    reply_id: str = ""
    author_display_name: str = ""
    like_count: int = Field(default=0, ge=0)
    published_at: datetime | None = None

    @field_validator("reply_id", "author_display_name", mode="before")
    @classmethod
    def normalize_nullable_reply_fields(cls, value: object) -> object:
        return "" if value is None else value


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
    comment_replies: list[ExportedReply] = Field(default_factory=list)

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


class TextUnit(_ContractModel):
    """One model input with an unambiguous comment/reply role."""

    record_id: str = Field(min_length=1)
    text: str
    text_kind: TextKind
    parent_record_id: str | None = None
    author: str = ""
    published_at: datetime | None = None
    like_count: int = Field(default=0, ge=0)
    video_id: str = ""
    video_title: str = ""
    video_channel: str = ""
    video_url: str = ""
    search_query: str = ""

    @model_validator(mode="after")
    def validate_parent(self) -> "TextUnit":
        if self.text_kind == TextKind.COMMENT and self.parent_record_id is not None:
            msg = "top-level comments cannot have parent_record_id"
            raise ValueError(msg)
        if self.text_kind == TextKind.REPLY and not self.parent_record_id:
            msg = "replies require parent_record_id"
            raise ValueError(msg)
        return self


class CleanedTextUnit(TextUnit):
    """A retained text unit enriched for downstream embedding."""

    clean_text: str = Field(min_length=1)
    detected_language: str
    duplicate_count: int = Field(default=1, ge=1)


class EmbeddingArtifactManifest(_ContractModel):
    """Alignment and reproducibility contract consumed by semantic deduplication."""

    schema_version: int = 1
    records_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    embeddings_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    n_records: int = Field(ge=0)
    dimensions: int = Field(ge=1)
    model_name: str = Field(min_length=1)
    model_revision: str | None = None
    normalized: bool
    record_ids_path: str | None = None
    record_ids_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    dtype: str = "float32"
    device: str = "unknown"
    batch_size: int | None = Field(default=None, ge=1)
    max_seq_length: int | None = Field(default=None, ge=1)
    prompt_prefix: str = ""
    config_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    created_at: datetime | None = None


def exported_comment_to_text_units(comment: ExportedComment) -> list[TextUnit]:
    """Flatten a thread without losing whether text belongs to a comment or reply."""
    canonical = comment.model_dump_json(exclude={"comment_replies"})
    comment_id = comment.comment_id.strip() or f"comment:{hashlib.sha256(canonical.encode()).hexdigest()}"
    common = {
        "video_id": comment.video_id,
        "video_title": comment.video_title,
        "video_channel": comment.video_channel,
        "video_url": comment.video_url,
        "search_query": comment.search_query,
    }
    units = [
        TextUnit(
            record_id=comment_id,
            text=comment.comment_text,
            text_kind=TextKind.COMMENT,
            author=comment.comment_author,
            published_at=comment.comment_published_at,
            like_count=comment.comment_like_count,
            **common,
        ),
    ]
    for index, reply in enumerate(comment.comment_replies):
        identity = f"{comment_id}\x1f{index}\x1f{reply.text}\x1f{reply.published_at}"
        reply_id = reply.reply_id.strip() or f"reply:{hashlib.sha256(identity.encode()).hexdigest()}"
        units.append(
            TextUnit(
                record_id=reply_id,
                text=reply.text,
                text_kind=TextKind.REPLY,
                parent_record_id=comment_id,
                author=reply.author_display_name,
                published_at=reply.published_at,
                like_count=reply.like_count,
                **common,
            ),
        )
    return units


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
