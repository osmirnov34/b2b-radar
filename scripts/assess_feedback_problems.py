#!/usr/bin/env python3
"""Generate conservative problem assessments for canonical feedback."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.operations.problem_assessment import assess_feedback_problems  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Assess dental feedback using versioned conservative rules.")
    parser.add_argument("feedback", type=Path)
    parser.add_argument("feedback_manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--taxonomy", type=Path, default=Path("configs/dental-problem-taxonomy.v1.json"))
    parser.add_argument("--rules", type=Path, default=Path("configs/dental-problem-rules.v1.json"))
    parser.add_argument("--topic-assignments", type=Path)
    parser.add_argument("--topic-manifest", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = assess_feedback_problems(
            args.feedback,
            args.feedback_manifest,
            args.taxonomy,
            args.rules,
            args.output,
            topic_assignments_path=args.topic_assignments,
            topic_manifest_path=args.topic_manifest,
        )
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Feedback: {result.feedback_records}; assessed: {result.assessed_feedback_records}")
    print(f"Assessments: {result.assessments}; hybrid: {result.hybrid_assessments}")
    print(f"Output: {result.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
