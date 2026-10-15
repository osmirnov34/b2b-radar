#!/usr/bin/env python3
"""Audit a Yandex Maps reviews JSONL without logging private values."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sources import YandexReviewAuditConfig, audit_yandex_reviews, write_yandex_review_audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a privacy-safe structural audit of Yandex reviews JSONL.")
    parser.add_argument("path", type=Path, help="Path to the immutable source JSONL.")
    parser.add_argument("--report-dir", type=Path, default=Path("data/reports/yandex-reviews"))
    parser.add_argument("--reference-date", type=date.fromisoformat)
    parser.add_argument("--minimum-reviews", type=int, default=30)
    parser.add_argument("--max-error-rate", type=float, default=0.01)
    parser.add_argument("--error-sample-limit", type=int, default=100)
    parser.add_argument("--top-organization-limit", type=int, default=20)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.path.is_file():
        parser.error(f"dataset file not found: {args.path}")
    try:
        report = audit_yandex_reviews(
            args.path,
            config=YandexReviewAuditConfig(
                reference_date=args.reference_date,
                minimum_reviews_for_comparison=args.minimum_reviews,
                max_error_rate=args.max_error_rate,
                error_sample_limit=args.error_sample_limit,
                top_organization_limit=args.top_organization_limit,
            ),
        )
    except ValueError as exc:
        parser.error(str(exc))
    json_path, markdown_path, errors_path = write_yandex_review_audit(report, args.report_dir)
    print(f"Status: {'usable' if report.is_usable else 'blocked'}")
    print(f"Records: {report.contract_valid}; invalid: {report.contract_invalid + report.json_invalid}")
    print(f"Organizations: {report.unique_organizations}")
    print(f"Publication range: {report.published_at_min} to {report.published_at_max}")
    print(f"Reports: {json_path}, {markdown_path}, {errors_path}")
    for warning in report.warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    if not report.is_usable:
        for message in report.critical_errors:
            print(f"ERROR: {message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
