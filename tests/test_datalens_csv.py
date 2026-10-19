"""Privacy, reconciliation and publication checks for file-based BI delivery."""

import csv
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.operations import datalens_csv as exporter
from tests.test_analytical_report import _report


def _sources() -> dict:
    report = _report()
    sources = {name: [] for name in exporter.COLUMNS}
    for index, org in enumerate(report.organizations):
        common = {
            "run_id": 1,
            "dataset_id": 1,
            "organization_id": org.organization_id,
            "organization_name": org.organization_name,
            "feedback_count": org.feedback_count,
            "sample_sufficient": False,
        }
        ratings = {"rating_sum": org.rating_sum, "low_rating_count": org.low_rating_count}
        distribution = {f"rating_{value}_count": 0 for value in range(1, 6)}
        distribution[f"rating_{1 if index == 0 else 5}_count"] = org.feedback_count
        month = report.months[0].month
        sources["organization_overview"].append({**common, **ratings, **distribution})
        sources["organization_monthly"].append({**common, **ratings, "month": month})
        category = {
            "category_id": "waiting_time",
            "category_name_ru": "Ожидание приёма",
            "category_active": True,
            "complaint_count": 2 if index == 0 else 0,
            "mixed_count": 0 if index == 0 else 1,
            "praise_count": 0 if index == 0 else 3,
            "accepted_complaint_count": 1 if index == 0 else 0,
            "unreviewed_complaint_count": 1 if index == 0 else 0,
        }
        sources["category_overview"].append(
            {
                **common,
                **category,
                "accepted_count": 1 if index == 0 else 0,
                "unreviewed_count": 1 if index == 0 else 4,
                "rejected_count": 1 if index == 0 else 0,
            }
        )
        sources["category_monthly"].append({**common, **category, "month": month})
    return sources


def _saved_report(directory: Path) -> None:
    directory.mkdir()
    report = _report()
    files = {"report.json": report.model_dump_json().encode(), "report.pdf": b"%PDF-test"}
    for name, content in files.items():
        (directory / name).write_bytes(content)
    manifest = {
        name: getattr(report, name) for name in ("run_id", "dataset_id", "dataset_sha256", "assessments_sha256")
    }
    manifest["files_sha256"] = {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}
    (directory / "manifest.json").write_text(json.dumps(manifest))


def _session(sources: dict) -> MagicMock:
    session = MagicMock(spec=AsyncSession)
    session.in_transaction.return_value = False
    results = [None, None, None]
    for name in exporter.COLUMNS:
        result = MagicMock()
        result.mappings.return_value = sources[name]
        results.append(result)
    session.execute = AsyncMock(side_effect=results)
    return session


def test_all_four_grains_reconcile_with_saved_report() -> None:
    exporter.validate_sources(_report(), _sources())


@pytest.mark.parametrize("mutation", ["duplicate", "private", "run", "month", "denominator", "signal", "rating"])
def test_invalid_aggregates_are_rejected(mutation: str) -> None:
    sources = deepcopy(_sources())
    if mutation == "duplicate":
        sources["organization_overview"].append(sources["organization_overview"][0].copy())
    elif mutation == "private":
        sources["organization_overview"][0]["text"] = "Private source text"
    elif mutation == "run":
        sources["category_overview"][0]["run_id"] = 2
    elif mutation == "month":
        sources["category_monthly"].pop()
    elif mutation == "denominator":
        sources["category_overview"][0]["feedback_count"] = 4
    elif mutation == "signal":
        sources["category_monthly"][0]["mixed_count"] = 1
    else:
        sources["organization_monthly"][0]["rating_sum"] += 1
    with pytest.raises(ValueError, match="do not reconcile") as captured:
        exporter.validate_sources(_report(), sources)
    assert "Private source text" not in str(captured.value)


async def test_export_checksums_readonly_and_no_overwrite(tmp_path: Path) -> None:
    report = tmp_path / "report"
    _saved_report(report)
    output = tmp_path / "csv"
    session = _session(_sources())
    manifest = await exporter.export_datalens_csv(session, report, output)
    assert "REPEATABLE READ, READ ONLY" in str(session.execute.call_args_list[0].args[0])
    assert str(session.execute.call_args_list[1].args[0]) == "SET LOCAL enable_mergejoin = off"
    assert "statement_timeout" in str(session.execute.call_args_list[2].args[0])
    assert manifest["rows"] == dict.fromkeys(exporter.COLUMNS, 2)
    for name, checksum in manifest["files_sha256"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == checksum
    assert output.stat().st_mode & 0o777 == 0o700
    for name in exporter.COLUMNS:
        with (output / f"{name}.csv").open(encoding="utf-8-sig", newline="") as source:
            rows = list(csv.DictReader(source))
        assert len(rows) == 2
        assert "text" not in rows[0]
    with pytest.raises(ValueError, match="new output directory"):
        await exporter.export_datalens_csv(_session(_sources()), report, output)


async def test_export_failure_does_not_publish_partial_package(tmp_path: Path) -> None:
    report = tmp_path / "report"
    _saved_report(report)
    sources = _sources()
    sources["category_monthly"].pop()
    output = tmp_path / "bad"
    with pytest.raises(ValueError, match="do not reconcile"):
        await exporter.export_datalens_csv(_session(sources), report, output)
    assert not output.exists()
    (report / "report.pdf").write_bytes(b"changed")
    with pytest.raises(ValueError, match="do not reconcile"):
        exporter.load_verified_report(report)


def test_csv_escaping_formula_safety_and_size_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "values.csv"
    exporter._write_csv(target, ("name",), [{"name": '  =HYPERLINK("bad")\n,текст'}])
    with target.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.reader(source))
    assert rows[1][0].startswith("'  =HYPERLINK")
    assert "\n,текст" in rows[1][0]
    monkeypatch.setattr(exporter, "_MAX_FILE_BYTES", 1)
    with pytest.raises(ValueError, match="size limit"):
        exporter._write_csv(tmp_path / "too-large.csv", ("name",), [{"name": "value"}])
