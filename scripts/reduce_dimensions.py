#!/usr/bin/env python3
"""Create UMAP spaces for clustering and visualization."""

# ruff: noqa: E402, T201 -- standalone CLI configures project imports and reports status.

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

from src.ml import ReductionMode, UMAPConfig, reduce_dimensions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reduce final corpus embeddings with reproducible UMAP models.")
    parser.add_argument("embeddings", type=Path, help="final-embeddings.npy from stage 7.")
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/umap.example.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/umap"))
    parser.add_argument("--mode", choices=("clustering", "visualization", "both"), default="both")
    parser.add_argument("--limit", type=int, help="Reduce only the first N rows for a disposable trial.")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.embeddings.is_file():
        parser.error(f"final embeddings file not found: {args.embeddings}")
    if not args.corpus_manifest.is_file():
        parser.error(f"corpus manifest file not found: {args.corpus_manifest}")
    if not args.config.is_file():
        parser.error(f"configuration file not found: {args.config}")
    modes = (
        (ReductionMode.CLUSTERING, ReductionMode.VISUALIZATION)
        if args.mode == "both"
        else (ReductionMode(args.mode),)
    )
    try:
        config = UMAPConfig.model_validate_json(args.config.read_text(encoding="utf-8"))
        manifests = reduce_dimensions(
            args.embeddings,
            args.corpus_manifest,
            args.output_dir,
            config=config,
            modes=modes,
            force=args.force,
            limit=args.limit,
        )
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    for mode, manifest in manifests.items():
        print(
            f"{mode.value}: {manifest.output_records} x {manifest.output_dimensions}; "
            f"trustworthiness={manifest.quality.trustworthiness}",
        )
        print(f"Manifest: {args.output_dir / f'{mode.value}-manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
