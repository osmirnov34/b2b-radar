"""Source-neutral contracts for the analytical product layer."""

from src.analytics.artifacts import (
    FeedbackMLAdaptationManifest,
    FeedbackPreparationManifest,
    FeedbackReconciliationReport,
)
from src.analytics.contracts import FeedbackRecord, FeedbackSource, FeedbackTopicAssignment
from src.analytics.io import FeedbackArtifactError, iter_feedback_records, parse_feedback_line
from src.analytics.problems import (
    AssessmentMethod,
    BusinessImpactHypothesis,
    EvidenceSpan,
    FeedbackProblemAssessment,
    FeedbackStance,
    ProblemAssessmentBatch,
    ProblemCategory,
    ProblemTaxonomy,
    ReviewStatus,
    load_problem_taxonomy,
    validate_assessment_batch,
)

__all__ = [
    "AssessmentMethod",
    "BusinessImpactHypothesis",
    "EvidenceSpan",
    "FeedbackArtifactError",
    "FeedbackMLAdaptationManifest",
    "FeedbackPreparationManifest",
    "FeedbackProblemAssessment",
    "FeedbackReconciliationReport",
    "FeedbackRecord",
    "FeedbackSource",
    "FeedbackStance",
    "FeedbackTopicAssignment",
    "ProblemAssessmentBatch",
    "ProblemCategory",
    "ProblemTaxonomy",
    "ReviewStatus",
    "iter_feedback_records",
    "load_problem_taxonomy",
    "parse_feedback_line",
    "validate_assessment_batch",
]
