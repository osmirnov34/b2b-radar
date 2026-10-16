#!/usr/bin/env python3
"""Compare a canonical feedback artifact with its PostgreSQL snapshot."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Sequence

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_settings
from src.operations.feedback_loading import AnalyticsLoadError
from src.operations.feedback_reconciliation import reconcile_feedback_snapshot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reconcile a canonical artifact with PostgreSQL.")
    parser.add_argument("artifact", type=Path, help="Canonical feedback JSONL.")
    parser.add_argument("manifest", type=Path, help="FeedbackPreparationManifest JSON.")
    parser.add_argument("--report", type=Path, help="Optional new path for the privacy-safe JSON report.")
    return parser


async def _reconcile(artifact: Path, manifest: Path) -> object:
    engine = create_async_engine(get_settings().database_url)
    try:
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        async with session_factory() as session:
            return await reconcile_feedback_snapshot(session, artifact, manifest)
    finally:
        await engine.dispose()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = asyncio.run(_reconcile(args.artifact, args.manifest))
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            with args.report.open("x", encoding="utf-8") as destination:
                destination.write(f"{report.model_dump_json(indent=2)}\n")
    except (AnalyticsLoadError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Status: {'valid' if report.is_valid else 'invalid'}")
    print(f"Dataset ID: {report.dataset_id}")
    print(f"Records: expected={report.expected_records}, database={report.database_records}")
    print(f"Organizations: expected={report.expected_organizations}, database={report.database_organizations}")
    print(f"Record differences: missing={report.missing_records}, unexpected={report.unexpected_records}, changed={report.changed_records}")
    for error in report.errors:
        print(f"ERROR: {error}", file=sys.stderr)
    return 0 if report.is_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
