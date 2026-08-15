#!/usr/bin/env python3
"""Inspect a comments JSONL without logging private comment values."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis import create_research_sample, inspect_comments_jsonl, write_inspection_reports


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate and profile an exported comments JSONL.")
    parser.add_argument("path", type=Path, help="Path to the downloaded comments JSONL.")
    parser.add_argument("--report-dir", type=Path, default=Path("data/reports"))
    parser.add_argument("--expected-records", type=int)
    parser.add_argument("--expected-records-tolerance", type=float, default=0.10)
    parser.add_argument("--max-error-rate", type=float, default=0.01)
    parser.add_argument("--error-sample-limit", type=int, default=100)
    parser.add_argument("--sample-size", type=int, default=0, help="Also write a deterministic research sample.")
    parser.add_argument("--sample-output", type=Path, default=Path("data/samples/comments-sample.jsonl"))
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--max-per-video", type=int, default=100)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.path.is_file():
        parser.error(f"dataset file not found: {args.path}")
    if args.error_sample_limit < 0:
        parser.error("--error-sample-limit cannot be negative")
    if args.sample_size < 0:
        parser.error("--sample-size cannot be negative")

    try:
        report = inspect_comments_jsonl(
            args.path,
            error_sample_limit=args.error_sample_limit,
            max_error_rate=args.max_error_rate,
            expected_records=args.expected_records,
            expected_records_tolerance=args.expected_records_tolerance,
        )
    except ValueError as exc:
        parser.error(str(exc))

    json_report, markdown_report, errors_report = write_inspection_reports(report, args.report_dir)
    print(f"Expected format: {report.format.expected}")
    print(f"Detected format: {report.format.detected}")
    print(f"Records: {report.contract_valid}; errors: {report.json_invalid + report.contract_invalid}")
    print(f"Reports: {json_report}, {markdown_report}, {errors_report}")

    if not report.is_usable:
        for message in report.critical_errors:
            print(f"ERROR: {message}", file=sys.stderr)
        if report.format.details:
            print(f"Format detail: {report.format.details}", file=sys.stderr)
        return 1

    if args.sample_size:
        metadata = create_research_sample(
            args.path,
            args.sample_output,
            sample_size=args.sample_size,
            seed=args.sample_seed,
            max_records_per_video=args.max_per_video,
        )
        print(f"Sample: {metadata.output_path} ({metadata.written_records} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
