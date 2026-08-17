#!/usr/bin/env python3
"""Group semantic near-duplicates from aligned cleaned records and embeddings."""

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

from src.ml import DeduplicationConfig, run_semantic_deduplication


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ANN semantic deduplication for cleaned text units.")
    parser.add_argument("records", type=Path, help="Path to development-clean.jsonl.")
    parser.add_argument("embeddings", type=Path, help="Path to aligned NumPy embeddings.")
    parser.add_argument("--embedding-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/semantic-deduplication.example.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/semantic-deduplication"))
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    for label, path in (
        ("cleaned records", args.records),
        ("embeddings", args.embeddings),
        ("embedding manifest", args.embedding_manifest),
        ("configuration", args.config),
    ):
        if not path.is_file():
            parser.error(f"{label} file not found: {path}")
    try:
        config = DeduplicationConfig.model_validate_json(args.config.read_text(encoding="utf-8"))
        manifest = run_semantic_deduplication(
            args.records,
            args.embeddings,
            args.embedding_manifest,
            args.output_dir,
            config=config,
            force=args.force,
        )
    except (FileExistsError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Input: {manifest.result.n_input}")
    print(f"Kept: {manifest.result.n_kept}; removed: {manifest.result.n_removed}")
    print(f"Manifest: {args.output_dir / 'semantic-deduplication-manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
