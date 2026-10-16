#!/usr/bin/env python3
"""Join an existing ML run to the canonical analytical feedback artifact."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.operations.feedback_ml_adaptation import adapt_feedback_ml_results  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Adapt checksum-verified ML topics to canonical feedback.")
    parser.add_argument("feedback", type=Path, help="Canonical feedback JSONL.")
    parser.add_argument("feedback_manifest", type=Path, help="Feedback preparation manifest.")
    parser.add_argument("corpus_manifest", type=Path, help="ML stage 07 corpus manifest.")
    parser.add_argument("reassignment_manifest", type=Path, help="ML stage 11 reassignment manifest.")
    parser.add_argument("output", type=Path, help="New privacy-safe topic assignment JSONL.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = adapt_feedback_ml_results(
            args.feedback,
            args.feedback_manifest,
            args.corpus_manifest,
            args.reassignment_manifest,
            args.output,
        )
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Matched records: {result.matched_records}/{result.feedback_records}")
    print(f"Assigned: {result.assigned_records}; outliers: {result.outlier_records}")
    print(f"Topics: {result.topics}; reassigned: {result.reassigned_records}")
    print(f"Output: {result.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
