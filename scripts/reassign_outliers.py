#!/usr/bin/env python3
"""Conservatively reassign high-confidence HDBSCAN outliers."""

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

from src.ml import OutlierReassignmentConfig, reassign_outliers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reassign only high-similarity, high-margin cluster outliers.")
    parser.add_argument("embeddings", type=Path, help="final-embeddings.npy from stage 7.")
    parser.add_argument("labels", type=Path, help="cluster-labels.npy from stage 9.")
    parser.add_argument("--probabilities", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--clustering-manifest", type=Path, required=True)
    parser.add_argument("--topic-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/outlier-reassignment.example.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/outlier-reassignment"))
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    for label, path in (
        ("final embeddings", args.embeddings),
        ("cluster labels", args.labels),
        ("cluster probabilities", args.probabilities),
        ("final corpus", args.corpus),
        ("corpus manifest", args.corpus_manifest),
        ("clustering manifest", args.clustering_manifest),
        ("topic manifest", args.topic_manifest),
        ("configuration", args.config),
    ):
        if not path.is_file():
            parser.error(f"{label} file not found: {path}")
    try:
        config = OutlierReassignmentConfig.model_validate_json(args.config.read_text(encoding="utf-8"))
        manifest = reassign_outliers(
            args.embeddings,
            args.labels,
            args.probabilities,
            args.corpus,
            args.corpus_manifest,
            args.clustering_manifest,
            args.topic_manifest,
            args.output_dir,
            config=config,
            force=args.force,
        )
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    metrics = manifest.metrics
    print(f"Original outliers: {metrics.original_outliers}")
    print(f"Reassigned: {metrics.reassigned_outliers}; remaining: {metrics.remaining_outliers}")
    print(f"Manifest: {args.output_dir / 'outlier-reassignment-manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
