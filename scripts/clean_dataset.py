#!/usr/bin/env python3
"""Clean checksum-verified development comments and replies."""

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

from src.ml import CleaningConfig, clean_development_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean only the checksum-verified development split.")
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path("data/interim/splits/development.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/interim/splits/split-manifest.json"),
    )
    parser.add_argument("--config", type=Path, default=Path("configs/dataset-cleaning.example.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/cleaning"))
    parser.add_argument("--force", action="store_true", help="Overwrite existing cleaning artifacts.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.path.is_file():
        parser.error(f"development dataset not found: {args.path}")
    if not args.manifest.is_file():
        parser.error(f"split manifest not found: {args.manifest}")
    if not args.config.is_file():
        parser.error(f"cleaning config not found: {args.config}")
    try:
        config = CleaningConfig.model_validate_json(args.config.read_text(encoding="utf-8"))
        result = clean_development_dataset(
            args.path,
            args.manifest,
            args.output_dir,
            config=config,
            force=args.force,
        )
    except (FileExistsError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    stats = result.stats
    print(f"Input: {stats.input_rows} rows, {stats.input_text_units} text units")
    print(f"Retained: {stats.output_text_units}; removed: {sum(stats.removed_by_reason.values())}")
    print(f"Comments/replies retained: {stats.output_comments}/{stats.output_replies}")
    print(f"Output: {result.output_path}")
    print(f"Manifest: {args.output_dir / 'cleaning-manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
