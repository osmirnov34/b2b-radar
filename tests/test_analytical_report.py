import hashlib
import importlib.util
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from pydantic import ValidationError
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from src.analytics.pdf_report import write_analytical_report
from src.analytics.report import AnalyticalReport
from src.operations.analytical_report import load_analytical_report
from src.operations.datalens_csv import export_datalens_csv

FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")


def _report() -> AnalyticalReport:
    return AnalyticalReport.model_validate(
        {
            "run_id": 1,
            "dataset_id": 1,
            "dataset_sha256": "a" * 64,
            "assessments_sha256": "b" * 64,
            "taxonomy_id": "dentistry",
            "taxonomy_version": "1.0.0",
            "ruleset_version": "1.0.0",
            "source": "yandex_maps",
            "run_generated_at": datetime(2026, 10, 16, tzinfo=UTC),
            "created_at": datetime(2026, 10, 19, tzinfo=UTC),
            "feedback_count": 10,
            "published_on_min": "2026-09-01",
            "published_on_max": "2026-09-30",
            "organizations": [
                {
                    "organization_id": "a",
                    "organization_name": "Клиника <b>без HTML</b> & " + "Длинное название " * 15,
                    "feedback_count": 3,
                    "rating_sum": 3,
                    "low_rating_count": 3,
                },
                {
                    "organization_id": "b",
                    "organization_name": "Вторая клиника",
                    "feedback_count": 7,
                    "rating_sum": 35,
                    "low_rating_count": 0,
                },
            ],
            "categories": [
                {
                    "category_id": "waiting_time",
                    "category_name_ru": "Ожидание приёма",
                    "feedback_count": 10,
                    "complaint_count": 2,
                    "mixed_count": 1,
                    "praise_count": 3,
                    "accepted_complaint_count": 1,
                    "unreviewed_complaint_count": 1,
                    "rejected_count": 1,
                }
            ],
            "months": [{"month": "2026-09-01", "feedback_count": 10, "rating_sum": 38}],
            "recommendations": [
                {
                    "organization_id": "a",
                    "category_name_ru": "Ожидание приёма",
                    "score": 20,
                    "requires_review": True,
                    "action_ru": "Проверить разметку.",
                    "explanation_ru": "Предварительная оценка.",
                }
            ],
        }
    )


def test_report_rejects_wrong_category_denominators() -> None:
    data = _report().model_dump()
    data["categories"][0]["feedback_count"] = 9
    with pytest.raises(ValidationError, match="denominators"):
        AnalyticalReport.model_validate(data)


def test_pdf_publishes_checksums_and_preserves_existing_result(tmp_path: Path) -> None:
    pytest.importorskip("reportlab")
    if not FONT.is_file():
        pytest.skip("Cyrillic font is unavailable")
    output = tmp_path / "report"
    manifest = write_analytical_report(_report(), output, FONT)
    assert (output / "report.pdf").read_bytes().startswith(b"%PDF-")
    for name, expected in manifest["files_sha256"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected
    json_output = (output / "report.json").read_text()
    assert "evidence_feedback_ids" not in json_output
    with pytest.raises(FileExistsError):
        write_analytical_report(_report(), output, FONT)


def test_pdf_text_contains_russian_weighted_numbers_and_escaped_names(tmp_path: Path) -> None:
    pytest.importorskip("reportlab")
    extractor = shutil.which("pdftotext")
    if extractor is None or not FONT.is_file():
        pytest.skip("PDF text extractor or Cyrillic font is unavailable")
    output = tmp_path / "report"
    write_analytical_report(_report(), output, FONT)
    extracted = subprocess.run(  # noqa: S603 -- fixed pdftotext executable and generated test PDF.
        [extractor, str(output / "report.pdf"), "-"], check=True, capture_output=True, text=True
    ).stdout
    assert "Аналитический отчёт" in extracted
    assert "3.80" in extracted
    assert "30.0%" in extracted
    assert "<b>без HTML</b>" in extracted
    assert "требуется ручная проверка" in extracted


def test_failed_render_leaves_no_published_directory(tmp_path: Path) -> None:
    pytest.importorskip("reportlab")
    output = tmp_path / "report"
    with pytest.raises(FileNotFoundError):
        write_analytical_report(_report(), output, tmp_path / "missing.ttf")
    assert not output.exists()


async def assert_report_snapshot(engine: AsyncEngine) -> None:
    """Exercise the report loader on the PostgreSQL regression fixture."""
    path = Path(__file__).parents[1] / "migrations/versions/e1f2a3b4c5d6_add_datalens_sources.py"
    spec = importlib.util.spec_from_file_location("report_bi_sources", path)
    assert spec is not None
    assert spec.loader is not None
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    async with engine.begin() as connection:
        await connection.exec_driver_sql("CREATE SCHEMA analytics_bi")
        await connection.exec_driver_sql("""
            UPDATE analytics.datasets SET published_on_min = '2026-09-01', published_on_max = '2026-09-01' WHERE id = 1
        """)
        for monthly in (False, True):
            for categories in (False, True):
                await connection.exec_driver_sql(migration.source_sql(monthly=monthly, categories=categories))
    try:
        async with async_sessionmaker(engine)() as session:
            report = await load_analytical_report(session, 1)
            assert not session.in_transaction()
        assert report.feedback_count == 3
        assert report.categories[0].complaint_count == 2
        assert report.categories[0].rejected_count == 1
        assert report.recommendations[0].requires_review
        assert "Private text" not in report.model_dump_json()
        if importlib.util.find_spec("reportlab") is not None and FONT.is_file():  # noqa: ASYNC240 -- local test fixture.
            with TemporaryDirectory() as temporary:
                root = Path(temporary)
                write_analytical_report(report, root / "pdf", FONT)
                async with async_sessionmaker(engine)() as session:
                    manifest = await export_datalens_csv(session, root / "pdf", root / "csv")
                    assert await session.scalar(text("SHOW enable_mergejoin")) == "on"
                assert manifest["rows"] == {
                    "organization_overview": 1,
                    "organization_monthly": 1,
                    "category_overview": 1,
                    "category_monthly": 1,
                }
                for file in (root / "csv").glob("*.csv"):
                    assert "Private text" not in file.read_text(encoding="utf-8-sig")
        assert "review-1" not in report.model_dump_json()
        async with async_sessionmaker(engine)() as session:
            with pytest.raises(ValueError, match="does not exist"):
                await load_analytical_report(session, 999)
    finally:
        async with engine.begin() as connection:
            await connection.exec_driver_sql("DROP SCHEMA analytics_bi CASCADE")
