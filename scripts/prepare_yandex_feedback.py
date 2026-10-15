#!/usr/bin/env python3
"""Prepare a private canonical feedback JSONL from Yandex Maps reviews."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.operations.feedback_preparation import FeedbackPreparationError, prepare_yandex_feedback


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare source-neutral analytical feedback JSONL.")
    parser.add_argument("source", type=Path, help="Verified Yandex Maps reviews JSONL.")
    parser.add_argument("output", type=Path, help="New private canonical JSONL path.")
    parser.add_argument(
        "--manifest",
        type=Path,
        help="New manifest path (default: <output>.manifest.json).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    manifest_path = args.manifest or args.output.with_suffix(f"{args.output.suffix}.manifest.json")
    try:
        manifest = prepare_yandex_feedback(args.source, args.output, manifest_path)
    except (FeedbackPreparationError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Records: {manifest.records}")
    print(f"Organizations: {manifest.unique_organizations}")
    print(f"Output: {args.output}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
