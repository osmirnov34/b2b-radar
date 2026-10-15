import json
from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.sources import (
    YandexReview,
    YandexReviewFormatError,
    YandexReviewLineError,
    iter_yandex_reviews,
    parse_yandex_review_line,
    yandex_review_to_feedback,
)


def _row(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "review_id": "7995cc14c8b92196",
        "review_text": " Исходный текст сохраняет пробелы ",
        "rating": 3,
        "author": "Автор",
        "published_at": "2025-02-06",
        "org_id": "229248496819",
        "org_name": "Стоматология",
        "org_url": "https://yandex.ru/maps/org/229248496819/reviews/",
        "search_query": "стоматология",
    }
    row.update(updates)
    return row


def test_yandex_review_preserves_text_and_validates_source_identity() -> None:
    review = YandexReview.model_validate(_row())

    assert review.review_text == " Исходный текст сохраняет пробелы "
    assert review.published_at == date(2025, 2, 6)
    assert review.rating == 3
    assert review.org_id == "229248496819"


def test_yandex_review_normalizes_only_nullable_optional_strings() -> None:
    review = YandexReview.model_validate(_row(author=None, search_query=None))

    assert review.author == ""
    assert review.search_query == ""


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"review_id": "not-an-id"}, "review_id"),
        ({"org_id": "clinic-1"}, "org_id"),
        ({"rating": "3"}, "rating"),
        ({"rating": 6}, "rating"),
        ({"published_at": "06.02.2025"}, "published_at"),
        ({"org_url": "https://example.com/organization"}, "org_url"),
        ({"org_url": "https://yandex.ru/maps/org/999/reviews/"}, "org_url does not contain org_id"),
        ({"unexpected": "field"}, "extra_forbidden"),
    ],
)
def test_yandex_review_rejects_invalid_source_values(updates: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        YandexReview.model_validate(_row(**updates))


def test_line_error_does_not_expose_review_or_author() -> None:
    private_text = "Private medical information"
    private_author = "Private Person"
    raw = json.dumps(
        _row(review_text=private_text, author=private_author, rating=8),
        ensure_ascii=False,
    ).encode()

    with pytest.raises(YandexReviewLineError) as captured:
        parse_yandex_review_line(raw, line_number=17)

    diagnostic = str(captured.value)
    assert "line 17" in diagnostic
    assert "content_hash=" in diagnostic
    assert private_text not in diagnostic
    assert private_author not in diagnostic
    assert captured.value.fields == tuple(sorted(_row()))


def test_reader_streams_jsonl_and_rejects_wrong_container(tmp_path: Path) -> None:
    source = tmp_path / "reviews.jsonl"
    rows = [_row(), _row(review_id="6529392e51d48c89")]
    source.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")

    assert [review.review_id for review in iter_yandex_reviews(source)] == [
        "7995cc14c8b92196",
        "6529392e51d48c89",
    ]

    html = tmp_path / "reviews.html"
    html.write_text("<!doctype html><title>Not a dataset</title>", encoding="utf-8")
    with pytest.raises(YandexReviewFormatError, match="detected html"):
        list(iter_yandex_reviews(html))


def test_yandex_review_converts_to_private_source_neutral_feedback() -> None:
    review = YandexReview.model_validate(_row())

    feedback = yandex_review_to_feedback(review)

    assert feedback.feedback_id == "yandex_maps:7995cc14c8b92196"
    assert feedback.source_record_id == review.review_id
    assert feedback.text == review.review_text
    assert feedback.published_on == review.published_at
    assert feedback.organization_id == review.org_id
    assert "author" not in feedback.model_dump()
