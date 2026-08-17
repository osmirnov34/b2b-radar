#!/usr/bin/env python3
"""Cluster the stage 8 clustering space with HDBSCAN."""

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

from src.ml import HDBSCANConfig, cluster_corpus


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cluster the verified UMAP clustering space with HDBSCAN.")
    parser.add_argument("reduced", type=Path, help="clustering-reduced.npy from stage 8.")
    parser.add_argument("--reduction-manifest", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/hdbscan.example.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/clustering"))
    parser.add_argument("--limit", type=int, help="Cluster only the first N rows for a disposable trial.")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    for label, path in (
        ("clustering matrix", args.reduced),
        ("reduction manifest", args.reduction_manifest),
        ("corpus manifest", args.corpus_manifest),
        ("configuration", args.config),
    ):
        if not path.is_file():
            parser.error(f"{label} file not found: {path}")
    try:
        config = HDBSCANConfig.model_validate_json(args.config.read_text(encoding="utf-8"))
        manifest = cluster_corpus(
            args.reduced,
            args.reduction_manifest,
            args.corpus_manifest,
            args.output_dir,
            config=config,
            force=args.force,
            limit=args.limit,
        )
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Records: {manifest.metrics.records}")
    print(
        f"Clusters: {manifest.metrics.clusters}; "
        f"outliers: {manifest.metrics.outliers} ({manifest.metrics.outlier_share:.1%})",
    )
    print(f"Manifest: {args.output_dir / 'clustering-manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
