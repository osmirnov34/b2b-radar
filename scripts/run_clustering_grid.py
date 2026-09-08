#!/usr/bin/env python3
"""Run or resume a source-bound HDBSCAN min-cluster-size grid."""

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

from src.ml import ClusteringGridConfig, run_clustering_grid


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare HDBSCAN min_cluster_size values on one fixed reduction.")
    parser.add_argument("reduced", type=Path, help="clustering-reduced.npy from stage 8")
    parser.add_argument("--reduction-manifest", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/clustering-grid.example.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.config.is_file():
        parser.error(f"configuration file not found: {args.config}")
    try:
        config = ClusteringGridConfig.model_validate_json(args.config.read_text(encoding="utf-8"))
        manifest = run_clustering_grid(
            args.reduced,
            args.reduction_manifest,
            args.corpus_manifest,
            args.output_dir,
            config=config,
        )
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    for result in manifest.results:
        metrics = result.metrics
        print(
            f"min_cluster_size={result.min_cluster_size}: clusters={metrics.clusters}, "
            f"outliers={metrics.outlier_share:.1%}, mean_probability={metrics.mean_probability:.3f}",
        )
    print(f"Manifest: {args.output_dir / 'clustering-grid-manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
