#!/usr/bin/env python3
"""Generate aligned multilingual embeddings from cleaned JSONL records."""

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

from src.ml import EmbeddingConfig, generate_embeddings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate multilingual embeddings for cleaned text units.")
    parser.add_argument("records", type=Path, help="Path to development-clean.jsonl.")
    parser.add_argument("--config", type=Path, default=Path("configs/embeddings.example.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/embeddings"))
    parser.add_argument("--resume", action="store_true", help="Continue a compatible interrupted run.")
    parser.add_argument("--force", action="store_true", help="Replace final and partial artifacts.")
    parser.add_argument("--limit", type=int, help="Embed only the first N records for a trial run.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.records.is_file():
        parser.error(f"cleaned records file not found: {args.records}")
    if not args.config.is_file():
        parser.error(f"configuration file not found: {args.config}")
    if args.resume and args.force:
        parser.error("--resume and --force are mutually exclusive")
    try:
        config = EmbeddingConfig.model_validate_json(args.config.read_text(encoding="utf-8"))
        manifest = generate_embeddings(
            args.records,
            args.output_dir,
            config=config,
            resume=args.resume,
            force=args.force,
            limit=args.limit,
        )
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Records: {manifest.n_records}; dimensions: {manifest.dimensions}")
    print(f"Model: {manifest.model_name}; device: {manifest.device}")
    print(f"Manifest: {args.output_dir / 'embedding-manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
