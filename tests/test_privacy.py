import json
from pathlib import Path

import pytest

from src.ml.privacy import (
    PrivacyAuditStatus,
    PrivacyFindingSeverity,
    PrivacyPolicy,
    audit_aggregate_report,
    spreadsheet_safe_value,
)


def test_spreadsheet_safe_value_neutralizes_formulas_but_keeps_numbers() -> None:
    assert spreadsheet_safe_value("=HYPERLINK('bad')") == "'=HYPERLINK('bad')"
    assert spreadsheet_safe_value(" +SUM(1,2)") == "' +SUM(1,2)"
    assert spreadsheet_safe_value("-cmd|' /C calc'!A0") == "'-cmd|' /C calc'!A0"
    assert spreadsheet_safe_value("-1.25") == "-1.25"
    assert spreadsheet_safe_value("ordinary topic") == "ordinary topic"


def test_privacy_audit_accepts_known_aggregate_files_and_warns_for_html(tmp_path: Path) -> None:
    report = tmp_path / "report"
    (report / "tables").mkdir(parents=True)
    (report / "figures").mkdir()
    (report / "summary.json").write_text('{"topics": 2}\n', encoding="utf-8")
    (report / "tables/topic-summary.csv").write_text("topic_id,name\n0,'=safe\n", encoding="utf-8")
    (report / "figures/topic-languages.html").write_text("<html></html>\n", encoding="utf-8")

    manifest = audit_aggregate_report(report)

    assert manifest.status == PrivacyAuditStatus.PASSED_WITH_WARNINGS
    assert manifest.files_checked == 3
    assert (report / "privacy-audit-manifest.json").is_file()
    assert any(finding.check == "html_structural_limit" for finding in manifest.findings)


@pytest.mark.parametrize(
    ("relative_path", "content", "expected_check"),
    [
        ("summary.json", json.dumps({"text": "private"}), "forbidden_json_fields"),
        ("tables/topic-summary.csv", "topic_id,author\n0,private\n", "forbidden_csv_columns"),
        ("tables/topic-summary.csv", "topic_id,name\n0,=unsafe()\n", "spreadsheet_formula_injection"),
        ("unexpected.json", "{}\n", "file_allowlist"),
    ],
)
def test_privacy_audit_blocks_unsafe_aggregate_outputs(
    tmp_path: Path,
    relative_path: str,
    content: str,
    expected_check: str,
) -> None:
    report = tmp_path / "report"
    path = report / relative_path
    path.parent.mkdir(parents=True)
    path.write_text(content, encoding="utf-8")

    manifest = audit_aggregate_report(report, require_pass=False)

    assert manifest.status == PrivacyAuditStatus.BLOCKED
    finding = next(finding for finding in manifest.findings if finding.check == expected_check)
    assert finding.severity == PrivacyFindingSeverity.BLOCKED
    assert "private" not in finding.details
    with pytest.raises(FileExistsError, match="already exists"):
        audit_aggregate_report(report)


def test_privacy_audit_rejects_symlinks_without_following_them(tmp_path: Path) -> None:
    report = tmp_path / "report"
    report.mkdir()
    private = tmp_path / "private.json"
    private.write_text('{"text": "secret"}\n', encoding="utf-8")
    (report / "summary.json").symlink_to(private)

    manifest = audit_aggregate_report(report, require_pass=False)

    assert manifest.status == PrivacyAuditStatus.BLOCKED
    assert manifest.findings[0].check == "no_symlinks"
    assert "secret" not in manifest.model_dump_json()


def test_privacy_audit_raises_after_writing_blocked_manifest(tmp_path: Path) -> None:
    report = tmp_path / "report"
    report.mkdir()
    (report / "unexpected.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="privacy audit blocked"):
        audit_aggregate_report(report)

    payload = json.loads((report / "privacy-audit-manifest.json").read_text(encoding="utf-8"))
    assert payload["status"] == "blocked"


def test_privacy_policy_cannot_remove_mandatory_fields_or_use_recursive_allowlist() -> None:
    with pytest.raises(ValueError, match="cannot remove mandatory fields"):
        PrivacyPolicy(forbidden_fields=("text",))
    with pytest.raises(ValueError, match="recursive wildcards"):
        PrivacyPolicy(allowed_files=("**/*",))


def test_privacy_audit_rejects_a_symlinked_report_root(tmp_path: Path) -> None:
    actual = tmp_path / "actual"
    actual.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(actual, target_is_directory=True)

    with pytest.raises(ValueError, match="cannot be a symbolic link"):
        audit_aggregate_report(linked)
