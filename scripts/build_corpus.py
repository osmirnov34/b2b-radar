#!/usr/bin/env python3
"""Build the final aligned corpus after semantic deduplication."""

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

from src.ml import build_final_corpus


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a checksum-verified corpus and aligned embedding matrix.")
    parser.add_argument("records", type=Path, help="Cleaned JSONL from stage 4.")
    parser.add_argument("embeddings", type=Path, help="Aligned embeddings.npy from stage 5.")
    parser.add_argument("keep_indices", type=Path, help="keep-indices.json from stage 6.")
    parser.add_argument("--cleaning-manifest", type=Path, required=True)
    parser.add_argument("--embedding-manifest", type=Path, required=True)
    parser.add_argument("--groups", type=Path, required=True, help="semantic-groups.jsonl from stage 6.")
    parser.add_argument("--deduplication-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/corpus"))
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    inputs = (
        ("cleaned records", args.records),
        ("embeddings", args.embeddings),
        ("keep indices", args.keep_indices),
        ("cleaning manifest", args.cleaning_manifest),
        ("embedding manifest", args.embedding_manifest),
        ("semantic groups", args.groups),
        ("deduplication manifest", args.deduplication_manifest),
    )
    for label, path in inputs:
        if not path.is_file():
            parser.error(f"{label} file not found: {path}")
    try:
        manifest = build_final_corpus(
            args.records,
            args.embeddings,
            args.keep_indices,
            cleaning_manifest_path=args.cleaning_manifest,
            embedding_manifest_path=args.embedding_manifest,
            groups_path=args.groups,
            deduplication_manifest_path=args.deduplication_manifest,
            output_dir=args.output_dir,
            force=args.force,
        )
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Input: {manifest.stats.input_records}")
    print(f"Output: {manifest.stats.output_records}; removed: {manifest.stats.removed_semantic_duplicates}")
    print(f"Manifest: {args.output_dir / 'corpus-manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
