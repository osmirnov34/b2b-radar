import json
from datetime import date
from pathlib import Path

from scripts.audit_yandex_reviews import main
from src.sources import (
    YandexReviewAudit,
    YandexReviewAuditConfig,
    audit_yandex_reviews,
    write_yandex_review_audit,
)


def _row(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "review_id": "7995cc14c8b92196",
        "review_text": "Подробный отзыв о качестве лечения без личных данных",
        "rating": 4,
        "author": "Private Person",
        "published_at": "2025-02-06",
        "org_id": "229248496819",
        "org_name": "Стоматология",
        "org_url": "https://yandex.ru/maps/org/229248496819/reviews/",
        "search_query": "стоматология",
    }
    row.update(updates)
    return row


def _write(path: Path, rows: list[object]) -> None:
    path.write_text("".join(f"{json.dumps(row, ensure_ascii=False)}\n" for row in rows), encoding="utf-8")


def test_audit_profiles_review_organization_rating_and_date_quality(tmp_path: Path) -> None:
    source = tmp_path / "reviews.jsonl"
    _write(
        source,
        [
            _row(),
            _row(
                review_id="6529392e51d48c89",
                rating=1,
                published_at="2025-03-01",
                org_id="2",
                org_name="Другая стоматология",
                org_url="https://yandex.ru/maps/org/2/reviews/",
            ),
            _row(review_id="6b1ed2d4a2fe2eb5", rating=5, published_at="2027-01-01", extra_field="value"),
        ],
    )

    report = audit_yandex_reviews(
        source,
        config=YandexReviewAuditConfig(
            reference_date=date(2026, 1, 1),
            minimum_reviews_for_comparison=2,
            max_error_rate=1,
        ),
    )

    assert report.is_usable
    assert report.contract_valid == 2
    assert report.contract_invalid == 1
    assert report.unique_review_ids == 2
    assert report.unique_organizations == 2
    assert report.rating_distribution == {1: 1, 4: 1, 5: 1}
    assert report.published_at_min == date(2025, 2, 6)
    assert report.published_at_max == date(2027, 1, 1)
    assert report.future_dates == 1
    assert report.organizations_at_or_above_minimum_reviews == 0
    assert report.unknown_fields == {"extra_field": 1}
    assert "rating values absent from dataset: [2, 3]" in report.warnings


def test_audit_blocks_conflicting_review_ids_without_exposing_values(tmp_path: Path) -> None:
    private_text = "Private medical detail that must not appear in reports"
    private_author = "Private Author"
    source = tmp_path / "reviews.jsonl"
    _write(
        source,
        [
            _row(review_text=private_text, author=private_author),
            _row(review_text="Different text", author=private_author),
            _row(review_id="6b1ed2d4a2fe2eb5", rating=8, review_text=private_text, author=private_author),
        ],
    )

    report = audit_yandex_reviews(
        source,
        config=YandexReviewAuditConfig(reference_date=date(2026, 1, 1)),
    )
    json_path, markdown_path, errors_path = write_yandex_review_audit(report, tmp_path / "reports")
    combined = "\n".join(
        path.read_text(encoding="utf-8") for path in (json_path, markdown_path, errors_path)
    )

    assert not report.is_usable
    assert report.duplicate_review_ids == 1
    assert report.conflicting_review_ids == 1
    assert report.contract_invalid == 1
    assert private_text not in combined
    assert private_author not in combined
    restored = YandexReviewAudit.model_validate_json(json_path.read_text(encoding="utf-8"))
    assert restored.source_sha256 == report.source_sha256


def test_audit_reports_identity_and_url_anomalies(tmp_path: Path) -> None:
    source = tmp_path / "reviews.jsonl"
    _write(
        source,
        [
            _row(),
            _row(review_id="6529392e51d48c89", org_name="Новое название"),
            _row(
                review_id="6b1ed2d4a2fe2eb5",
                org_id="1104602194",
                org_name="Стоматология",
                org_url="https://yandex.ru/maps/org/1104602194/reviews/",
            ),
            _row(
                review_id="849a25d513520060",
                org_id="123",
                org_name="Некорректная ссылка",
                org_url="http://example.test/not-yandex",
            ),
        ],
    )

    report = audit_yandex_reviews(
        source,
        config=YandexReviewAuditConfig(reference_date=date(2026, 1, 1)),
    )

    assert report.organizations_with_multiple_names == 1
    assert report.organization_names_with_multiple_ids == 1
    assert report.invalid_organization_urls == 1
    assert report.warnings


def test_audit_cli_writes_all_reports(tmp_path: Path) -> None:
    source = tmp_path / "reviews.jsonl"
    report_dir = tmp_path / "reports"
    _write(source, [_row()])

    exit_code = main(
        [
            str(source),
            "--report-dir",
            str(report_dir),
            "--reference-date",
            "2026-01-01",
        ],
    )

    assert exit_code == 0
    assert (report_dir / "yandex-reviews-audit.json").is_file()
    assert (report_dir / "yandex-reviews-audit.md").is_file()
    assert (report_dir / "yandex-reviews-errors.jsonl").is_file()
