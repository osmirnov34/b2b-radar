from datetime import date

import pytest
from pydantic import ValidationError

from src.analytics import FeedbackRecord, FeedbackSource


def _feedback() -> FeedbackRecord:
    return FeedbackRecord(
        feedback_id="yandex_maps:7995cc14c8b92196",
        source=FeedbackSource.YANDEX_MAPS,
        source_record_id="7995cc14c8b92196",
        text="Врач подробно объяснил лечение",
        rating=4,
        published_on=date(2025, 6, 1),
        organization_id="123456789",
        organization_name="Стоматология",
        organization_url="https://yandex.ru/maps/org/123456789/",
        search_query="стоматология москва",
    )


def test_feedback_record_is_source_neutral_and_serializable() -> None:
    record = _feedback()

    assert record.source == FeedbackSource.YANDEX_MAPS
    assert record.model_dump(mode="json")["published_on"] == "2025-06-01"
    assert "author" not in type(record).model_fields


def test_feedback_record_is_strict_and_immutable() -> None:
    record = _feedback()

    with pytest.raises(ValidationError, match="frozen"):
        record.rating = 1
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        FeedbackRecord.model_validate({**record.model_dump(), "author": "private"})


@pytest.mark.parametrize("rating", [0, 6, "5"])
def test_feedback_record_rejects_invalid_or_coerced_rating(rating: object) -> None:
    with pytest.raises(ValidationError):
        FeedbackRecord.model_validate({**_feedback().model_dump(), "rating": rating})
