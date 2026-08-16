#!/usr/bin/env python3
"""Run privacy-safe EDA on a checksum-verified development split."""

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

from src.analysis import EDAConfig, profile_development_dataset, write_eda_reports


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile only the development comment split.")
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
    parser.add_argument("--report-dir", type=Path, default=Path("data/reports/eda"))
    parser.add_argument("--language-sample-size", type=int, default=20_000)
    parser.add_argument("--language-seed", type=int, default=42)
    parser.add_argument("--top-groups", type=int, default=20)
    parser.add_argument("--force", action="store_true", help="Overwrite existing EDA reports.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.path.is_file():
        parser.error(f"development dataset not found: {args.path}")
    if not args.manifest.is_file():
        parser.error(f"split manifest not found: {args.manifest}")
    try:
        config = EDAConfig(
            language_sample_size=args.language_sample_size,
            language_seed=args.language_seed,
            top_groups=args.top_groups,
        )
        profile = profile_development_dataset(args.path, args.manifest, config=config)
        paths = write_eda_reports(profile, args.report_dir, force=args.force)
    except (FileExistsError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Profiled {profile.records} development records")
    print(f"Language sample: {profile.language_sample_records}")
    print(f"Reports: {args.report_dir} ({len(paths)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
