"""Source-neutral contracts for the analytical product layer."""

from src.analytics.artifacts import (
    FeedbackMLAdaptationManifest,
    FeedbackPreparationManifest,
    FeedbackReconciliationReport,
    ProblemAssessmentArtifactManifest,
)
from src.analytics.contracts import FeedbackRecord, FeedbackSource, FeedbackTopicAssignment
from src.analytics.io import FeedbackArtifactError, iter_feedback_records, parse_feedback_line
from src.analytics.problems import (
    AssessmentMethod,
    BusinessImpactHypothesis,
    CategoryDetectionRule,
    EvidenceSpan,
    FeedbackProblemAssessment,
    FeedbackStance,
    ProblemAssessmentBatch,
    ProblemCategory,
    ProblemDetectionRules,
    ProblemTaxonomy,
    ReviewStatus,
    load_problem_detection_rules,
    load_problem_taxonomy,
    validate_assessment_batch,
    validate_problem_detection_rules,
)

__all__ = [
    "AssessmentMethod",
    "BusinessImpactHypothesis",
    "CategoryDetectionRule",
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
    "ProblemAssessmentArtifactManifest",
    "ProblemAssessmentBatch",
    "ProblemCategory",
    "ProblemDetectionRules",
    "ProblemTaxonomy",
    "ReviewStatus",
    "iter_feedback_records",
    "load_problem_detection_rules",
    "load_problem_taxonomy",
    "parse_feedback_line",
    "validate_assessment_batch",
    "validate_problem_detection_rules",
]
