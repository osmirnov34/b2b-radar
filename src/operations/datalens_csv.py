"""Export allowlisted aggregate snapshots for the file-based DataLens connector."""

# ruff: noqa: ASYNC240 -- Dedicated CLI export owns synchronous local file publication.

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from sqlalchemy import text

from src.analytics.report import AnalyticalReport
from src.operations.datalens_package import DashboardSpec

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from sqlalchemy.ext.asyncio import AsyncSession

_COMMON = ("run_id", "dataset_id", "organization_id", "organization_name", "feedback_count", "sample_sufficient")
_SIGNALS = ("complaint_count", "mixed_count", "praise_count", "accepted_complaint_count", "unreviewed_complaint_count")
_QUALITY = ("unreviewed_count", "accepted_count", "rejected_count")
_CATEGORY = ("category_id", "category_name_ru", "category_active", *_SIGNALS)
_RATINGS = ("rating_sum", "low_rating_count")
_DISTRIBUTION = tuple(f"rating_{value}_count" for value in range(1, 6))
COLUMNS = {
    "organization_overview": (*_COMMON, *_RATINGS, *_DISTRIBUTION),
    "organization_monthly": (*_COMMON, "month", *_RATINGS),
    "category_overview": (*_COMMON, *_CATEGORY, *_QUALITY),
    "category_monthly": (*_COMMON, "month", *_CATEGORY),
}
_SPEC_PATH = Path(__file__).resolve().parents[2] / "configs/datalens-dashboard.v1.json"
_MAX_FILE_BYTES = 190 * 1024 * 1024
_MINIMUM_SAMPLE = 30


def _require(condition: bool) -> None:
    if not condition:
        msg = "aggregate CSV sources do not reconcile with the selected PDF report"
        raise ValueError(msg)


def _equal(left: float, right: float) -> bool:
    return math.isfinite(left) and math.isfinite(right) and math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-7)


def load_verified_report(directory: Path) -> AnalyticalReport:
    """Verify the saved JSON/PDF pair before using its aggregates as acceptance targets."""
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for name in ("report.json", "report.pdf"):
        actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        _require(actual == manifest["files_sha256"][name])
    report = AnalyticalReport.model_validate_json((directory / "report.json").read_text(encoding="utf-8"))
    for name in ("run_id", "dataset_id", "dataset_sha256", "assessments_sha256"):
        _require(getattr(report, name) == manifest[name])
    return report


def validate_sources(report: AnalyticalReport, sources: dict[str, list[dict[str, Any]]]) -> None:  # noqa: C901, PLR0912, PLR0915
    """Reconcile grains, denominators, monthly signals and saved PDF aggregates."""
    _require(set(sources) == set(COLUMNS))
    organizations = {row.organization_id: row for row in report.organizations}
    categories = {row.category_id: row for row in report.categories}
    for source, rows in sources.items():
        keys = set()
        for row in rows:
            _require(set(row) == set(COLUMNS[source]))
            _require(row["run_id"] == report.run_id and row["dataset_id"] == report.dataset_id)
            _require(row["organization_id"] in organizations)
            _require(row["organization_name"] == organizations[row["organization_id"]].organization_name)
            grain = [row["organization_id"]]
            if "month" in row:
                _require(isinstance(row["month"], date))
                grain.append(row["month"])
            if "category_id" in row:
                _require(row["category_id"] in categories)
                _require(row["category_name_ru"] == categories[row["category_id"]].category_name_ru)
                _require(isinstance(row["category_active"], bool))
                grain.append(row["category_id"])
                _require(row["complaint_count"] == row["accepted_complaint_count"] + row["unreviewed_complaint_count"])
                _require(sum(row[name] for name in _SIGNALS[:3]) <= row["feedback_count"])
            for name, value in row.items():
                if name.endswith("_count") or name == "rating_sum":
                    _require(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0)
            _require(row["feedback_count"] > 0)
            _require(isinstance(row["sample_sufficient"], bool))
            _require(row["sample_sufficient"] == (row["feedback_count"] >= _MINIMUM_SAMPLE))
            key = tuple(grain)
            _require(key not in keys)
            keys.add(key)
    overview = {row["organization_id"]: row for row in sources["organization_overview"]}
    _require(set(overview) == set(organizations))
    for identifier, expected in organizations.items():
        row = overview[identifier]
        for name in ("feedback_count", *_RATINGS):
            _require(_equal(row[name], getattr(expected, name)))
        _require(sum(row[name] for name in _DISTRIBUTION) == row["feedback_count"])
        _require(_equal(sum(row[name] for name in _DISTRIBUTION[:3]), row["low_rating_count"]))
    monthly = {(row["organization_id"], row["month"]): row for row in sources["organization_monthly"]}
    by_month: dict[date, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    by_org: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for (identifier, month), row in monthly.items():
        for name in ("feedback_count", *_RATINGS):
            by_org[identifier][name] += row[name]
            by_month[month][name] += row[name]
    _require(set(by_org) == set(organizations) and set(by_month) == {row.month for row in report.months})
    for identifier, org_totals in by_org.items():
        for name in ("feedback_count", *_RATINGS):
            _require(_equal(org_totals[name], overview[identifier][name]))
    for month_row in report.months:
        for name in ("feedback_count", "rating_sum"):
            _require(_equal(by_month[month_row.month][name], getattr(month_row, name)))
    category_rows = {(row["organization_id"], row["category_id"]): row for row in sources["category_overview"]}
    _require(len(category_rows) == len(organizations) * len(categories))
    totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for (identifier, category), row in category_rows.items():
        _require(row["feedback_count"] == overview[identifier]["feedback_count"])
        for name in ("feedback_count", *_SIGNALS, "rejected_count"):
            totals[category][name] += row[name]
    for category, expected_category in categories.items():
        for name in ("feedback_count", *_SIGNALS, "rejected_count"):
            _require(_equal(totals[category][name], getattr(expected_category, name)))
    monthly_signals: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    _require(len(sources["category_monthly"]) == len(monthly) * len(categories))
    for row in sources["category_monthly"]:
        key = (row["organization_id"], row["month"])
        _require(key in monthly and row["feedback_count"] == monthly[key]["feedback_count"])
        for name in _SIGNALS:
            monthly_signals[(row["organization_id"], row["category_id"])][name] += row[name]
    for key, row in category_rows.items():
        for name in _SIGNALS:
            _require(monthly_signals[key][name] == row[name])


def _csv_value(value: object) -> object:
    """Keep dates/booleans portable and neutralize spreadsheet formulas in text cells."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def _write_csv(path: Path, columns: Sequence[str], rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8-sig", newline="") as target:
        writer = csv.writer(target, quoting=csv.QUOTE_ALL)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([_csv_value(row[name]) for name in columns])
            if target.tell() > _MAX_FILE_BYTES:
                msg = "CSV exceeds the conservative DataLens file size limit"
                raise ValueError(msg)


async def export_datalens_csv(session: AsyncSession, report_dir: Path, output_dir: Path) -> dict[str, object]:
    """Read one repeatable-read snapshot and publish reconciled, aggregate-only CSV files."""
    if output_dir.exists() or session.in_transaction():
        msg = "CSV export requires a new output directory and a fresh database session"
        raise ValueError(msg)
    report = load_verified_report(report_dir)
    spec = DashboardSpec.model_validate_json(_SPEC_PATH.read_text(encoding="utf-8"))
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".datalens-csv-", dir=output_dir.parent) as temporary:
        staging = Path(temporary)
        sources = {}
        async with session.begin():
            await session.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
            # Monthly aggregate cardinalities are underestimated: a merge on category alone
            # compares millions of unrelated organization/month pairs. Scope this planner
            # choice to this bulk export only, never to the server or application sessions.
            await session.execute(text("SET LOCAL enable_mergejoin = off"))
            await session.execute(text("SET LOCAL statement_timeout = '60s'"))
            for source, columns in COLUMNS.items():
                order = ", ".join(spec.datasets[source].grain)
                query = f"SELECT {', '.join(columns)} FROM analytics_bi.{source} WHERE run_id=:run ORDER BY {order}"  # noqa: S608 -- fixed column/source allowlists.
                result = await session.execute(text(query), {"run": report.run_id})
                rows = [dict(row) for row in result.mappings()]
                _write_csv(staging / f"{source}.csv", columns, rows)
                sources[source] = rows
            validate_sources(report, sources)
        (staging / "dashboard.json").write_text(spec.model_dump_json(indent=2) + "\n", encoding="utf-8")
        (staging / "README.md").write_text(
            f"# DataLens CSV\n\nRun ID: {report.run_id}; dataset ID: {report.dataset_id}.\n\n"
            "Upload only the four CSV files via the Files connector. Encoding: UTF-8 with BOM; "
            "delimiter: comma; quote: double quote; header: yes.\n"
            "IDs must be strings, month a date, counts integers and sums fractional numbers. "
            "Keep sample_sufficient/category_active boolean.\n"
            "Do not join the four tables. Category denominators repeat per category.\n"
            "Potential spreadsheet formulas in text are prefixed with an apostrophe.\n"
            "dashboard.json is an assembly specification, not an import file. "
            "See docs/datalens-csv.md in the repository. Do not publish the dashboard without review.\n",
            encoding="utf-8",
        )
        manifest: dict[str, object] = {
            "schema_version": 1,
            "classification": "aggregate_datalens_csv",
            "status": "reconciled_with_saved_report",
            "transaction_settings": {"enable_mergejoin": "off", "statement_timeout": "60s", "read_only": True},
            "run_id": report.run_id,
            "dataset_id": report.dataset_id,
            "dataset_sha256": report.dataset_sha256,
            "assessments_sha256": report.assessments_sha256,
            "report_json_sha256": hashlib.sha256((report_dir / "report.json").read_bytes()).hexdigest(),
            "report_pdf_sha256": hashlib.sha256((report_dir / "report.pdf").read_bytes()).hexdigest(),
            "rows": {name: len(rows) for name, rows in sources.items()},
            "files_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in staging.iterdir()},
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        # Reserve the target exclusively; publish the manifest last as the completion marker.
        output_dir.mkdir(mode=0o700)
        for path in sorted(staging.iterdir(), key=lambda path: path.name == "manifest.json"):
            os.link(path, output_dir / path.name)
    return manifest
