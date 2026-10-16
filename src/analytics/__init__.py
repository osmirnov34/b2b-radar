"""Source-neutral contracts for the analytical product layer."""

from src.analytics.artifacts import FeedbackPreparationManifest
from src.analytics.contracts import FeedbackRecord, FeedbackSource
from src.analytics.io import FeedbackArtifactError, iter_feedback_records, parse_feedback_line

__all__ = [
    "FeedbackArtifactError",
    "FeedbackPreparationManifest",
    "FeedbackRecord",
    "FeedbackSource",
    "iter_feedback_records",
    "parse_feedback_line",
]
