from pydantic import BaseModel, Field


class FilterSettings(BaseModel):
    """Quality threshold configuration (>= semantics; 0 disables)."""

    # Source quality gates.
    source_min_views: int = Field(default=0, ge=0)
    source_min_likes: int = Field(default=0, ge=0)
    source_min_comments: int = Field(default=0, ge=0)
    source_min_duration_seconds: int = Field(default=0, ge=0)
    source_max_age_days: int = Field(default=0, ge=0)

    # Document quality gates.
    document_min_likes: int = Field(default=0, ge=0)
    document_min_length: int = Field(default=0, ge=0)
    document_min_replies: int = Field(default=0, ge=0)
