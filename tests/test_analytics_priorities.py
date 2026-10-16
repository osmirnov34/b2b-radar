import importlib.util
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.analytics.priorities import ProblemPriorityInput, rank_problem_priorities


def _signal(**updates: object) -> ProblemPriorityInput:
    data: dict[str, object] = {
        "run_id": 1,
        "dataset_id": 1,
        "organization_id": "clinic",
        "category_id": "waiting_time",
        "category_name_ru": "Ожидание",
        "feedback_count": 100,
        "complaint_count": 10,
        "accepted_complaint_count": 10,
        "unreviewed_complaint_count": 0,
        "mean_category_probability": 0.8,
    }
    return ProblemPriorityInput.model_validate({**data, **updates})


def test_rejected_only_signal_cannot_create_priority() -> None:
    signal = _signal(complaint_count=0, accepted_complaint_count=0, mean_category_probability=None)
    assert rank_problem_priorities([signal]) == []


def test_unreviewed_and_small_samples_require_review() -> None:
    pending = _signal(accepted_complaint_count=0, unreviewed_complaint_count=10)
    small = _signal(feedback_count=20)
    for signal in (pending, small):
        result = rank_problem_priorities([signal])[0]
        assert result.status == "review_required"
        assert "Проверить разметку" in result.recommendation_ru
    assert rank_problem_priorities([_signal()])[0].status == "investigate"


def test_trend_requires_two_sufficient_periods_and_score_is_explainable() -> None:
    trend = {
        "previous_complaint_share": 0.1,
        "current_complaint_share": 0.2,
        "previous_feedback_count": 30,
        "current_feedback_count": 30,
    }
    result = rank_problem_priorities([_signal(**trend)])[0]
    assert result.trend_available
    assert result.trend_component == pytest.approx(10)
    assert result.score == pytest.approx(
        sum(
            (result.prevalence_component, result.volume_component, result.confidence_component, result.trend_component)
        ),
        abs=0.0001,
    )
    trend["current_feedback_count"] = 29
    assert rank_problem_priorities([_signal(**trend)])[0].trend_component == 0


def test_invalid_review_counts_are_rejected() -> None:
    with pytest.raises(ValidationError, match="accepted plus unreviewed"):
        _signal(accepted_complaint_count=9)


def test_ranking_is_deterministic_and_increases_with_prevalence() -> None:
    baseline = _signal(organization_id="baseline")
    higher = _signal(organization_id="higher", complaint_count=20, accepted_complaint_count=20)
    assert rank_problem_priorities([baseline, higher])[0].signal.organization_id == "higher"
    assert rank_problem_priorities([baseline, higher]) == rank_problem_priorities([higher, baseline])


def _migration() -> object:
    path = Path(__file__).parents[1] / "migrations/versions/d0e1f2a3b4c5_fix_reviewed_problem_kpis.py"
    spec = importlib.util.spec_from_file_location("reviewed_kpis", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_corrected_sql_filters_signals_but_retains_rejection_quality_counts() -> None:
    module = _migration()
    for monthly in (False, True):
        sql = module._view_sql(monthly=monthly, corrected=True)
        assert "pa.stance = 'complaint' AND pa.review_status <> 'rejected'" in sql
        assert "accepted_complaint_share" in sql
        assert "unreviewed_complaint_share" in sql
        assert "NULLIF(totals.feedback_count, 0)" in sql
    assert "WHERE pa.review_status = 'rejected'" in module._view_sql(monthly=False, corrected=True)
