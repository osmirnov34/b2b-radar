"""Privacy controls for aggregate ML reports and spreadsheet exports."""

from __future__ import annotations

import csv
import json
import re
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, ConfigDict, Field, model_validator

_NEGATIVE_NUMBER = re.compile(r"^-\d+(\.\d+)?$")
_MANDATORY_PRIVATE_FIELDS = frozenset(
    {
        "author",
        "author_display_name",
        "clean_text",
        "comment_author",
        "comment_id",
        "comment_text",
        "parent_record_id",
        "record_id",
        "search_query",
        "text",
        "video_id",
        "video_url",
    },
)


class _PrivacyModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PrivacyAuditStatus(StrEnum):
    """Summarize whether an aggregate report passed privacy controls."""

    PASSED = "passed"
    PASSED_WITH_WARNINGS = "passed_with_warnings"
    BLOCKED = "blocked"


class PrivacyFindingSeverity(StrEnum):
    """Classify a privacy audit result without copying offending values."""

    INFO = "info"
    WARNING = "warning"
    BLOCKED = "blocked"


class PrivacyPolicy(_PrivacyModel):
    """Allow known aggregate outputs and reject source-level identifiers."""

    schema_version: int = 1
    forbidden_fields: tuple[str, ...] = (
        "author",
        "author_display_name",
        "clean_text",
        "comment_author",
        "comment_id",
        "comment_text",
        "parent_record_id",
        "record_id",
        "search_query",
        "text",
        "video_id",
        "video_url",
    )
    allowed_files: tuple[str, ...] = (
        "analysis-summary.json",
        "cluster-cards/topic-*.json",
        "figures/cluster-overview.png",
        "figures/dataset-splits.html",
        "figures/development-processing-flow.html",
        "figures/problem-review-priority.html",
        "figures/topic-*-languages.html",
        "figures/topic-languages.html",
        "figures/topic-problem-signals.html",
        "figures/topic-similarity.html",
        "figures/topic-temporal-trends.html",
        "figures/umap-clusters.html",
        "figures/video-concentration.html",
        "manual-topic-review.csv",
        "privacy-audit-manifest.json",
        "problem-priorities-manifest.json",
        "problem-priorities.jsonl",
        "problem-signals-manifest.json",
        "problem-signals.jsonl",
        "report-manifest.json",
        "review/manual-topic-review.csv",
        "summary.json",
        "tables/source-concentration.csv",
        "tables/topic-summary.csv",
        "topic-temporal-trends-manifest.json",
        "topic-temporal-trends.jsonl",
        "video-concentration-manifest.json",
        "video-concentration.jsonl",
    )
    html_requires_warning: bool = True

    @model_validator(mode="after")
    def protect_policy_invariants(self) -> PrivacyPolicy:
        normalized_fields = {field.casefold() for field in self.forbidden_fields}
        missing = sorted(_MANDATORY_PRIVATE_FIELDS - normalized_fields)
        if missing:
            msg = f"privacy policy cannot remove mandatory fields: {', '.join(missing)}"
            raise ValueError(msg)
        if not self.allowed_files or len(set(self.allowed_files)) != len(self.allowed_files):
            msg = "privacy report allowlist must be non-empty and contain no duplicates"
            raise ValueError(msg)
        if any("**" in pattern for pattern in self.allowed_files):
            msg = "privacy report allowlist cannot contain recursive wildcards"
            raise ValueError(msg)
        return self


class PrivacyFinding(_PrivacyModel):
    """Record file-level evidence without storing a sensitive value."""

    severity: PrivacyFindingSeverity
    path: str
    check: str
    details: str


class PrivacyAuditManifest(_PrivacyModel):
    """Persist the policy and outcome of an aggregate-report privacy audit."""

    audit_schema_version: int = 1
    report_dir: str
    policy: PrivacyPolicy
    status: PrivacyAuditStatus
    files_checked: int = Field(ge=0)
    findings: list[PrivacyFinding]
    audited_at: datetime


def spreadsheet_safe_value(value: str) -> str:
    """Neutralize spreadsheet formulas while preserving ordinary and numeric text."""
    stripped = value.lstrip()
    dangerous = stripped.startswith(("=", "+", "@", "\t", "\r")) or (
        stripped.startswith("-") and _NEGATIVE_NUMBER.fullmatch(stripped) is None
    )
    return f"'{value}" if dangerous else value


def _allowed(relative: str, policy: PrivacyPolicy) -> bool:
    path = PurePosixPath(relative)
    return any(path.match(pattern) for pattern in policy.allowed_files)


def _forbidden_json_keys(value: object, forbidden: set[str]) -> set[str]:
    if isinstance(value, dict):
        found = {str(key).casefold() for key in value if str(key).casefold() in forbidden}
        for nested in value.values():
            found.update(_forbidden_json_keys(nested, forbidden))
        return found
    if isinstance(value, list):
        list_found: set[str] = set()
        for nested in value:
            list_found.update(_forbidden_json_keys(nested, forbidden))
        return list_found
    return set()


def _audit_json(path: Path, relative: str, forbidden: set[str], *, json_lines: bool) -> list[PrivacyFinding]:
    findings = []
    try:
        with path.open(encoding="utf-8") as source:
            values = [json.loads(line) for line in source if line.strip()] if json_lines else [json.load(source)]
        keys: set[str] = set()
        for value in values:
            keys.update(_forbidden_json_keys(value, forbidden))
        if keys:
            findings.append(
                PrivacyFinding(
                    severity=PrivacyFindingSeverity.BLOCKED,
                    path=relative,
                    check="forbidden_json_fields",
                    details=f"forbidden field names found: {', '.join(sorted(keys))}",
                ),
            )
    except (OSError, json.JSONDecodeError) as exc:
        findings.append(
            PrivacyFinding(
                severity=PrivacyFindingSeverity.BLOCKED,
                path=relative,
                check="structured_file_readable",
                details=f"cannot parse structured aggregate file: {type(exc).__name__}",
            ),
        )
    return findings


def _audit_csv(path: Path, relative: str, forbidden: set[str]) -> list[PrivacyFinding]:
    findings = []
    try:
        with path.open(encoding="utf-8", newline="") as source:
            reader = csv.reader(source)
            header = next(reader, [])
            forbidden_headers = sorted({name.casefold() for name in header} & forbidden)
            unsafe_cells = sum(
                1
                for row in reader
                for value in row
                if spreadsheet_safe_value(value) != value
            )
        if forbidden_headers:
            findings.append(
                PrivacyFinding(
                    severity=PrivacyFindingSeverity.BLOCKED,
                    path=relative,
                    check="forbidden_csv_columns",
                    details=f"forbidden columns found: {', '.join(forbidden_headers)}",
                ),
            )
        if unsafe_cells:
            findings.append(
                PrivacyFinding(
                    severity=PrivacyFindingSeverity.BLOCKED,
                    path=relative,
                    check="spreadsheet_formula_injection",
                    details=f"unescaped formula-like cells found: {unsafe_cells}",
                ),
            )
    except (OSError, csv.Error) as exc:
        findings.append(
            PrivacyFinding(
                severity=PrivacyFindingSeverity.BLOCKED,
                path=relative,
                check="structured_file_readable",
                details=f"cannot parse aggregate CSV: {type(exc).__name__}",
            ),
        )
    return findings


def _audit_file(path: Path, root: Path, policy: PrivacyPolicy, forbidden: set[str]) -> list[PrivacyFinding]:
    relative = path.relative_to(root).as_posix()
    if path.is_symlink():
        findings = [
            PrivacyFinding(
                severity=PrivacyFindingSeverity.BLOCKED,
                path=relative,
                check="no_symlinks",
                details="symbolic links are not allowed in aggregate reports",
            ),
        ]
    elif not _allowed(relative, policy):
        findings = [
            PrivacyFinding(
                severity=PrivacyFindingSeverity.BLOCKED,
                path=relative,
                check="file_allowlist",
                details="file is not declared by the aggregate-report policy",
            ),
        ]
    elif path.suffix.casefold() == ".json":
        findings = _audit_json(path, relative, forbidden, json_lines=False)
    elif path.suffix.casefold() == ".jsonl":
        findings = _audit_json(path, relative, forbidden, json_lines=True)
    elif path.suffix.casefold() == ".csv":
        findings = _audit_csv(path, relative, forbidden)
    elif path.suffix.casefold() == ".html" and policy.html_requires_warning:
        findings = [
            PrivacyFinding(
                severity=PrivacyFindingSeverity.WARNING,
                path=relative,
                check="html_structural_limit",
                details="known generated HTML is allowlisted but not structurally privacy-inspected",
            ),
        ]
    else:
        findings = []
    return findings


def audit_aggregate_report(
    report_dir: Path,
    *,
    policy: PrivacyPolicy | None = None,
    overwrite: bool = False,
    require_pass: bool = True,
) -> PrivacyAuditManifest:
    """Audit and manifest a report directory without reading source ML artifacts."""
    active_policy = policy or PrivacyPolicy()
    if report_dir.is_symlink():
        msg = f"aggregate report directory cannot be a symbolic link: {report_dir}"
        raise ValueError(msg)
    root = report_dir.resolve()
    if not root.is_dir():
        msg = f"aggregate report directory does not exist: {root}"
        raise FileNotFoundError(msg)
    manifest_path = root / "privacy-audit-manifest.json"
    if manifest_path.exists() and not overwrite:
        msg = f"privacy audit manifest already exists: {manifest_path}"
        raise FileExistsError(msg)
    findings: list[PrivacyFinding] = []
    files = sorted(path for path in root.rglob("*") if path.is_file() or path.is_symlink())
    forbidden = {field.casefold() for field in active_policy.forbidden_fields}
    for path in files:
        findings.extend(_audit_file(path, root, active_policy, forbidden))
    blocked = any(finding.severity == PrivacyFindingSeverity.BLOCKED for finding in findings)
    warned = any(finding.severity == PrivacyFindingSeverity.WARNING for finding in findings)
    status = (
        PrivacyAuditStatus.BLOCKED
        if blocked
        else PrivacyAuditStatus.PASSED_WITH_WARNINGS
        if warned
        else PrivacyAuditStatus.PASSED
    )
    manifest = PrivacyAuditManifest(
        report_dir=str(root),
        policy=active_policy,
        status=status,
        files_checked=len(files),
        findings=findings,
        audited_at=datetime.now(UTC),
    )
    manifest_tmp = manifest_path.with_name(f".{manifest_path.name}.tmp")
    manifest_tmp.write_text(f"{manifest.model_dump_json(indent=2)}\n", encoding="utf-8")
    manifest_tmp.replace(manifest_path)
    if blocked and require_pass:
        msg = f"aggregate report privacy audit blocked; review {manifest_path}"
        raise ValueError(msg)
    return manifest
