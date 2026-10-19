"""Russian PDF rendering from validated aggregate report contracts."""

# ruff: noqa: PLC0415 -- ReportLab is an optional dependency, loaded only for PDF rendering.
# ruff: noqa: RUF001 -- Russian report text intentionally contains Cyrillic and typographic punctuation.
from __future__ import annotations

import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING
from xml.sax.saxutils import escape

if TYPE_CHECKING:
    from collections.abc import Sequence

    from reportlab.graphics.shapes import Drawing
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.platypus import BaseDocTemplate, Flowable, Paragraph

    from src.analytics.report import AnalyticalReport

_MINIMUM_SAMPLE = 30


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _register_font(path: Path) -> str:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    name = "Report-" + _sha256(path)[:16]
    font = TTFont(name, str(path))
    if not all(ord(letter) in font.face.charToGlyph for letter in "Аналитический отчётЁё"):
        msg = "report font must contain Cyrillic glyphs"
        raise ValueError(msg)
    pdfmetrics.registerFont(font)
    return name


def _paragraph(value: str, style: ParagraphStyle) -> Paragraph:
    from reportlab.platypus import Paragraph

    return Paragraph(escape(value), style)


def _bars(rows: Sequence[tuple[str, float]], font: str, label: str) -> Drawing:
    from reportlab.graphics.shapes import Drawing, Rect, String
    from reportlab.lib.colors import HexColor

    height = max(len(rows), 1) * 22 + 28
    drawing = Drawing(480, height)
    drawing.add(String(0, height - 12, label, fontName=font, fontSize=8))
    for index, (name, value) in enumerate(rows):
        y = height - 36 - index * 22
        drawing.add(String(0, y, name[:35], fontName=font, fontSize=7))
        drawing.add(Rect(180, y - 2, max(value, 0) * 2.5, 10, fillColor=HexColor("#256080"), strokeColor=None))
        drawing.add(String(435, y, f"{value:.1f}%", fontName=font, fontSize=8))
    return drawing


def _table(rows: list[list[str]], style: ParagraphStyle) -> Flowable:
    from reportlab.lib.colors import HexColor
    from reportlab.platypus import LongTable, TableStyle

    table = LongTable(
        [[_paragraph(value, style) for value in row] for row in rows], repeatRows=1, colWidths=[150, 110, 110, 110]
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), HexColor("#e4edf3")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def render_analytical_pdf(report: AnalyticalReport, output: Path, font_path: Path) -> None:
    """Render text and vector charts without accessing any source comments or database."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import PageBreak, SimpleDocTemplate, Spacer

    font = _register_font(font_path)
    body = ParagraphStyle("body", fontName=font, fontSize=9, leading=13, spaceAfter=8)
    heading = ParagraphStyle("heading", parent=body, fontSize=15, leading=20, spaceBefore=12)
    total = report.feedback_count
    rating = sum(row.rating_sum for row in report.organizations) / total
    low = sum(row.low_rating_count for row in report.organizations) / total
    small = sum(row.feedback_count < _MINIMUM_SAMPLE for row in report.organizations)
    story: list[Flowable] = [
        _paragraph("Аналитический отчёт: стоматологии Москвы", heading),
        _paragraph(f"Запуск {report.run_id} · набор данных {report.dataset_id} · источник {report.source}", body),
        _paragraph(
            f"Период отзывов: {report.published_on_min} — {report.published_on_max}. "
            f"Подготовлено: {report.created_at.isoformat()}.",
            body,
        ),
        _paragraph("1. Основные показатели", heading),
        _paragraph(
            f"Отзывов: {total}. Клиник: {len(report.organizations)}. Средний рейтинг: {rating:.2f}. "
            f"Доля оценок 1–3: {low:.1%}. Клиник с выборкой меньше 30: {small}.",
            body,
        ),
        _paragraph(
            "Все показатели описывают доступную выборку отзывов, а не весь рынок Москвы. "
            "Средний рейтинг взвешен по числу отзывов. Низкий рейтинг сам по себе не устанавливает причину жалобы.",
            body,
        ),
    ]
    strongest = sorted(report.categories, key=lambda row: (-row.complaint_count, row.category_id))
    if strongest[0].complaint_count:
        top = strongest[0]
        story.append(
            _paragraph(
                f"Наибольшая доля жалоб: {top.category_name_ru} — "
                f"{top.complaint_count} из {total} ({top.complaint_count / total:.1%}).",
                body,
            )
        )
    else:
        story.append(_paragraph("В этом запуске нет неотклонённых оценок с тональностью «жалоба».", body))
    story.extend(
        [
            _paragraph("2. Проблемы и сильные стороны", heading),
            _paragraph(
                "Знаменатель — все отзывы выборки для каждой категории. Один отзыв может относиться "
                "к нескольким категориям: доли категорий нельзя складывать. Отклонённая разметка исключена.",
                body,
            ),
        ]
    )
    for title, field in (
        ("Жалобы", "complaint_count"),
        ("Сильные стороны", "praise_count"),
        ("Смешанные оценки", "mixed_count"),
    ):
        rows = sorted(report.categories, key=lambda row: (-getattr(row, field), row.category_id))
        story.extend(
            [
                _bars([(row.category_name_ru, getattr(row, field) / total * 100) for row in rows], font, title),
                Spacer(1, 8),
            ]
        )
    story.extend([PageBreak(), _paragraph("3. Клиники и динамика", heading)])
    organizations = sorted(report.organizations, key=lambda row: (-row.feedback_count, row.organization_id))
    story.append(
        _paragraph(
            "Показаны 20 клиник с наибольшим объёмом отзывов; все агрегаты доступны в report.json. "
            "Порог 30 — ориентир достаточности объёма, а не гарантия статистической точности.",
            body,
        )
    )
    table = [["Клиника / ID", "Отзывов", "Средний рейтинг", "Выборка ≥ 30"]]
    table.extend(
        [
            f"{row.organization_name} / {row.organization_id}",
            str(row.feedback_count),
            f"{row.rating_sum / row.feedback_count:.2f}",
            "да" if row.feedback_count >= _MINIMUM_SAMPLE else "нет",
        ]
        for row in organizations[:20]
    )
    story.append(_table(table, body))
    monthly = [["Месяц", "Отзывов", "Средний рейтинг", "Выборка ≥ 30"]]
    monthly.extend(
        [
            row.month.isoformat(),
            str(row.feedback_count),
            f"{row.rating_sum / row.feedback_count:.2f}",
            "да" if row.feedback_count >= _MINIMUM_SAMPLE else "нет",
        ]
        for row in report.months
    )
    story.extend(
        [
            _paragraph("Месячные показатели", heading),
            _table(monthly, body),
            _paragraph(
                "Месяцы без исходных отзывов пропущены; это не нулевая активность. "
                "Текущий месяц может быть неполным. Динамика рекомендаций использует только два завершённых "
                "месяца относительно даты запуска и требует по 30 отзывов в каждом.",
                body,
            ),
            PageBreak(),
            _paragraph("4. Приоритеты и рекомендуемые проверки", heading),
            _paragraph(
                "Баллы определяют очередь исследования, не финансовые потери. "
                "Компоненты: доля жалоб до 60, объём до 15, уверенность до 15, рост доли до 10. "
                "Уверенность базовой разметки — эвристика. Показаны первые 10 приоритетов.",
                body,
            ),
        ]
    )
    names = {row.organization_id: row.organization_name for row in report.organizations}
    for index, recommendation in enumerate(report.recommendations[:10], start=1):
        status = "требуется ручная проверка" if recommendation.requires_review else "проверить причину"
        story.extend(
            [
                _paragraph(
                    f"{index}. {names.get(recommendation.organization_id, recommendation.organization_id)} / "
                    f"{recommendation.category_name_ru} — {recommendation.score:.1f} балла; {status}.",
                    heading,
                ),
                _paragraph(recommendation.action_ru, body),
                _paragraph(recommendation.explanation_ru, body),
            ]
        )
    if not report.recommendations:
        story.append(_paragraph("Приоритетов нет: отсутствие сигнала не доказывает отсутствие проблемы.", body))
    accepted = sum(row.accepted_complaint_count for row in report.categories)
    unreviewed = sum(row.unreviewed_complaint_count for row in report.categories)
    rejected = sum(row.rejected_count for row in report.categories)
    story.extend(
        [
            _paragraph("5. Качество, происхождение и ограничения", heading),
            _paragraph(
                f"Категорий: {len(report.categories)}. Принятых жалоб-категорий: {accepted}; "
                f"предварительных: {unreviewed}; отклонённых оценок всех тональностей: {rejected}. "
                "Это числа оценок, а не уникальных людей или отзывов.",
                body,
            ),
            _paragraph(
                "Выборка может быть смещена, в том числе по рейтингу и способу сбора. "
                "Предварительные оценки требуют проверки. Отчёт не устанавливает медицинские ошибки, "
                "не подтверждает причинность и не оценивает окупаемость или потери дохода. "
                "Отсутствие жалоб не равно доказанному качеству услуги. Тексты отзывов и авторы не включены.",
                body,
            ),
            _paragraph(
                f"Таксономия: {report.taxonomy_id} {report.taxonomy_version}; "
                f"правила: {report.ruleset_version}; дата запуска: {report.run_generated_at.isoformat()}.",
                body,
            ),
            _paragraph(f"SHA-256 данных: {report.dataset_sha256}", body),
            _paragraph(f"SHA-256 разметки: {report.assessments_sha256}", body),
        ]
    )
    document = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        leftMargin=42,
        rightMargin=42,
        topMargin=36,
        bottomMargin=36,
        title="Аналитический отчёт стоматологий Москвы",
    )
    def footer(canvas: Canvas, document: BaseDocTemplate) -> None:
        canvas.saveState()
        canvas.setFont(font, 7)
        canvas.drawString(42, 20, f"Запуск {report.run_id} · страница {document.page}")
        canvas.restoreState()

    document.build(story, onFirstPage=footer, onLaterPages=footer)


def write_analytical_report(report: AnalyticalReport, output_dir: Path, font_path: Path) -> dict[str, object]:
    """Atomically publish PDF, aggregate JSON and checksums in a new directory."""
    if output_dir.exists():
        msg = f"refusing to overwrite report directory: {output_dir}"
        raise FileExistsError(msg)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".pdf-report-", dir=output_dir.parent) as temporary:
        staging = Path(temporary) / "report"
        staging.mkdir()
        render_analytical_pdf(report, staging / "report.pdf", font_path)
        (staging / "report.json").write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
        manifest: dict[str, object] = {
            "schema_version": 1,
            "classification": "aggregate_management_report",
            "run_id": report.run_id,
            "dataset_id": report.dataset_id,
            "dataset_sha256": report.dataset_sha256,
            "assessments_sha256": report.assessments_sha256,
            "reportlab_version": version("reportlab"),
            "font_sha256": _sha256(font_path),
            "files_sha256": {name: _sha256(staging / name) for name in ("report.pdf", "report.json")},
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        staging.rename(output_dir)
    return manifest
