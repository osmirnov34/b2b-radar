from pydantic import BaseModel, Field


class FilterSettings(BaseModel):
    """Quality threshold configuration (>= semantics; 0 disables)."""

    # Document quality gates.
    document_min_likes: int = Field(default=0, ge=0)
    document_min_length: int = Field(default=0, ge=0)
    document_min_replies: int = Field(default=0, ge=0)
