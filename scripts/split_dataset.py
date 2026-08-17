#!/usr/bin/env python3
"""Create deterministic development, validation, and test comment datasets."""

# ruff: noqa: E402, T201 -- standalone CLI configures the project path and writes status to stdio.

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml import SplitConfig, SplitName, split_comments_jsonl, write_split_markdown


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Split an inspected comments JSONL by video/source group.")
    parser.add_argument("path", type=Path, help="Path to a valid comments JSONL.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/interim/splits"))
    parser.add_argument("--development-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-informative-chars", type=int, default=20)
    parser.add_argument("--min-informative-tokens", type=int, default=4)
    parser.add_argument("--force", action="store_true", help="Overwrite existing split outputs.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.path.is_file():
        parser.error(f"dataset file not found: {args.path}")

    try:
        config = SplitConfig(
            development_ratio=args.development_ratio,
            validation_ratio=args.validation_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            min_informative_chars=args.min_informative_chars,
            min_informative_tokens=args.min_informative_tokens,
        )
        manifest = split_comments_jsonl(args.path, args.output_dir, config=config, force=args.force)
        report_path = write_split_markdown(manifest, args.output_dir)
    except (FileExistsError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Source: {manifest.stats.input_records} records in {manifest.stats.input_groups} groups")
    for split in SplitName:
        print(
            f"{split.value}: assigned={manifest.stats.assigned_records.get(split, 0)}, "
            f"written={manifest.stats.written_records.get(split, 0)}, "
            f"leaks_removed={manifest.stats.removed_content_leaks.get(split, 0)}",
        )
    print(f"Manifest: {args.output_dir / 'split-manifest.json'}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
