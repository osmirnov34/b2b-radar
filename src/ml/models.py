from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.ml.schemas import ClusterComment, ExportedComment


class _InternalModel(BaseModel):
    """Strict value objects passed between offline-analysis stages."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class CommentRecord(_InternalModel):
    text: str
    author: str = ""
    channel: str = ""
    query: str = ""
    video_id: str = ""
    video_title: str = ""
    source_url: str = ""

    @classmethod
    def from_export(cls, row: ExportedComment) -> "CommentRecord":
        return cls(
            text=row.comment_text.strip(),
            author=row.comment_author,
            channel=row.video_channel,
            query=row.search_query,
            video_id=row.video_id,
            video_title=row.video_title,
            source_url=row.video_url,
        )

    @property
    def video_url(self) -> str:
        if self.source_url:
            return self.source_url
        return f"https://www.youtube.com/watch?v={self.video_id}" if self.video_id else ""

    @property
    def normalized_text_key(self) -> str:
        return " ".join(self.text.casefold().split())

    def to_cluster_comment(self) -> ClusterComment:
        return ClusterComment(
            text=self.text,
            author=self.author,
            channel=self.channel,
            query=self.query,
            video_id=self.video_id,
            video_title=self.video_title,
            video_url=self.video_url,
        )


class CleaningReason(StrEnum):
    EMPTY = "empty"
    TOO_SHORT = "too_short"
    TOO_LONG = "too_long"
    NO_ALPHANUMERIC = "no_alphanumeric"
    ACKNOWLEDGEMENT = "acknowledgement"
    FOREIGN_LANGUAGE = "foreign_language"
    DISALLOWED_LANGUAGE = "disallowed_language"
    PIPELINE_NOISE = "pipeline_noise"
    URL_ONLY = "url_only"
    REPEATED_CHARACTERS = "repeated_characters"
    MOSTLY_UPPERCASE = "mostly_uppercase"
    EXACT_DUPLICATE = "exact_duplicate"


class CleaningDecision(_InternalModel):
    keep: bool
    reason: CleaningReason | None = None

    @model_validator(mode="after")
    def validate_reason(self) -> "CleaningDecision":
        if self.keep and self.reason is not None:
            msg = "kept comments cannot have a removal reason"
            raise ValueError(msg)
        if not self.keep and self.reason is None:
            msg = "removed comments require a reason"
            raise ValueError(msg)
        return self


class CleaningStats(_InternalModel):
    n_input: int = Field(ge=0)
    n_kept: int = Field(ge=0)
    removed_by_reason: dict[CleaningReason, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_counts(self) -> "CleaningStats":
        if any(count < 0 for count in self.removed_by_reason.values()):
            msg = "removed counts cannot be negative"
            raise ValueError(msg)
        if self.n_kept + sum(self.removed_by_reason.values()) != self.n_input:
            msg = "cleaning counts must add up to n_input"
            raise ValueError(msg)
        return self


class CleaningResult(_InternalModel):
    comments: list[CommentRecord]
    stats: CleaningStats

    @model_validator(mode="after")
    def validate_kept_count(self) -> "CleaningResult":
        if len(self.comments) != self.stats.n_kept:
            msg = "comments length must match cleaning n_kept"
            raise ValueError(msg)
        return self


class DuplicatePair(_InternalModel):
    representative_index: int = Field(ge=0)
    duplicate_index: int = Field(ge=0)
    similarity: float | None = Field(default=None, ge=-1, le=1)

    @model_validator(mode="after")
    def validate_distinct_indices(self) -> "DuplicatePair":
        if self.representative_index == self.duplicate_index:
            msg = "representative and duplicate indices must differ"
            raise ValueError(msg)
        return self


class DuplicateGroup(_InternalModel):
    representative_index: int = Field(ge=0)
    duplicate_indices: list[int] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_indices(self) -> "DuplicateGroup":
        if any(index < 0 for index in self.duplicate_indices):
            msg = "duplicate indices cannot be negative"
            raise ValueError(msg)
        if len(set(self.duplicate_indices)) != len(self.duplicate_indices):
            msg = "duplicate indices must be unique"
            raise ValueError(msg)
        if self.representative_index in self.duplicate_indices:
            msg = "representative index cannot also be a duplicate"
            raise ValueError(msg)
        return self


class DeduplicationStats(_InternalModel):
    n_input: int = Field(ge=0)
    n_kept: int = Field(ge=0)
    n_removed: int = Field(ge=0)
    threshold: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_counts(self) -> "DeduplicationStats":
        if self.n_kept + self.n_removed != self.n_input:
            msg = "deduplication counts must add up to n_input"
            raise ValueError(msg)
        return self


class DeduplicationResult(_InternalModel):
    keep_indices: list[int]
    sample_pairs: list[DuplicatePair] = Field(default_factory=list)
    groups: list[DuplicateGroup] = Field(default_factory=list)
    stats: DeduplicationStats

    @model_validator(mode="after")
    def validate_keep_indices(self) -> "DeduplicationResult":
        if any(index < 0 for index in self.keep_indices):
            msg = "keep indices cannot be negative"
            raise ValueError(msg)
        if len(set(self.keep_indices)) != len(self.keep_indices):
            msg = "keep indices must be unique"
            raise ValueError(msg)
        if len(self.keep_indices) != self.stats.n_kept:
            msg = "keep indices length must match deduplication n_kept"
            raise ValueError(msg)
        return self
