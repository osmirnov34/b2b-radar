"""Explainable prioritization of aggregate problem signals."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

_MINIMUM_SAMPLE = 30

_ACTIONS = {
    "access_and_scheduling": "Проверить доступность записи, пропущенные звонки и причины отмен.",
    "waiting_time": "Измерить задержки приёмов и проверить планирование расписания.",
    "staff_communication": "Разобрать обращения и проверить стандарты общения персонала.",
    "treatment_quality": "Передать обращения ответственному за качество для проверки обстоятельств лечения.",
    "pain_and_comfort": "Проверить обратную связь пациентов относительно комфорта и качества информирования.",
    "pricing_and_billing": "Сверить предварительные сметы, согласование изменений и итоговые счета.",
    "treatment_trust": "Проверить понятность объяснений плана лечения и согласование процедур.",
    "cleanliness_and_safety": "Передать обращения ответственному за санитарный контроль для проверки.",
    "follow_up_support": "Проверить выдачу рекомендаций и доступность связи после лечения.",
    "facility_access": "Проверить инструкции проезда, навигацию и доступность входа.",
}


class ProblemPriorityInput(BaseModel):
    """One organization/category aggregate from the corrected KPI view."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    run_id: int = Field(ge=1)
    dataset_id: int = Field(ge=1)
    organization_id: str
    category_id: str
    category_name_ru: str
    feedback_count: int = Field(gt=0)
    complaint_count: int = Field(ge=0)
    accepted_complaint_count: int = Field(ge=0)
    unreviewed_complaint_count: int = Field(ge=0)
    mean_category_probability: float | None = Field(default=None, ge=0, le=1)
    previous_complaint_share: float | None = Field(default=None, ge=0, le=1)
    current_complaint_share: float | None = Field(default=None, ge=0, le=1)
    previous_feedback_count: int = Field(default=0, ge=0)
    current_feedback_count: int = Field(default=0, ge=0)
    evidence_feedback_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_counts(self) -> ProblemPriorityInput:
        """Reject inconsistent counters before producing management advice."""
        if self.complaint_count != self.accepted_complaint_count + self.unreviewed_complaint_count:
            msg = "complaint count must equal accepted plus unreviewed complaints"
            raise ValueError(msg)
        if self.complaint_count > self.feedback_count:
            msg = "complaint count cannot exceed source feedback count"
            raise ValueError(msg)
        if (self.previous_complaint_share is None) != (self.current_complaint_share is None):
            msg = "trend requires both comparison periods"
            raise ValueError(msg)
        return self


class ProblemPriority(BaseModel):
    """Relative investigation priority with transparent score components."""

    model_config = ConfigDict(frozen=True)

    signal: ProblemPriorityInput
    score: float = Field(ge=0, le=100)
    prevalence_component: float
    volume_component: float
    confidence_component: float
    trend_component: float
    trend_available: bool
    sample_sufficient: bool
    status: Literal["review_required", "investigate"]
    recommendation_ru: str
    explanation_ru: str


def rank_problem_priorities(signals: list[ProblemPriorityInput]) -> list[ProblemPriority]:
    """Rank positive complaint signals; scores express investigation priority, not revenue loss."""
    results: list[ProblemPriority] = []
    for signal in signals:
        if not signal.complaint_count:
            continue
        trend_available = (
            signal.previous_complaint_share is not None
            and signal.current_complaint_share is not None
            and min(signal.previous_feedback_count, signal.current_feedback_count) >= _MINIMUM_SAMPLE
        )
        delta = (
            float(signal.current_complaint_share or 0) - float(signal.previous_complaint_share or 0)
            if trend_available
            else 0.0
        )
        prevalence = 60 * signal.complaint_count / signal.feedback_count
        volume = 15 * min(math.log1p(signal.complaint_count) / math.log1p(100), 1)
        confidence = 15 * (signal.mean_category_probability or 0)
        trend = 10 * min(max(delta, 0) / 0.1, 1)
        sufficient = signal.feedback_count >= _MINIMUM_SAMPLE
        needs_review = signal.unreviewed_complaint_count > 0 or not sufficient
        action = _ACTIONS.get(signal.category_id, "Разобрать подтверждающие обращения и определить причину жалоб.")
        recommendation = "Проверить разметку и достаточность данных; затем: " + action if needs_review else action
        results.append(
            ProblemPriority(
                signal=signal,
                score=round(prevalence + volume + confidence + trend, 4),
                prevalence_component=prevalence,
                volume_component=volume,
                confidence_component=confidence,
                trend_component=trend,
                trend_available=trend_available,
                sample_sufficient=sufficient,
                status="review_required" if needs_review else "investigate",
                recommendation_ru=recommendation,
                explanation_ru=(
                    f"Жалобы: {signal.complaint_count} из {signal.feedback_count}; "
                    f"подтверждено: {signal.accepted_complaint_count}; "
                    f"ожидает проверки: {signal.unreviewed_complaint_count}. "
                    "Баллы: доля 60, объём 15, уверенность 15, рост доли 10. "
                    + ("Динамика учтена." if trend_available else "Для динамики недостаточно сопоставимых данных.")
                ),
            )
        )
    return sorted(results, key=lambda row: (-row.score, row.signal.organization_id, row.signal.category_id))
