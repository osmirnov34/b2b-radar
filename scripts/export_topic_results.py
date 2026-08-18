#!/usr/bin/env python3
"""Export fixed topic-analysis results for application and research consumers."""

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

from src.ml import ExportConfig, export_topic_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export safe topic catalog, assignments, and quality metadata.")
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--clustering-manifest", type=Path, required=True)
    parser.add_argument("--topic-manifest", type=Path, required=True)
    parser.add_argument("--reassignment-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/export.example.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/export"))
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    for label, path in (
        ("corpus manifest", args.corpus_manifest),
        ("clustering manifest", args.clustering_manifest),
        ("topic manifest", args.topic_manifest),
        ("reassignment manifest", args.reassignment_manifest),
        ("evaluation manifest", args.evaluation_manifest),
        ("configuration", args.config),
    ):
        if not path.is_file():
            parser.error(f"{label} file not found: {path}")
    try:
        config = ExportConfig.model_validate_json(args.config.read_text(encoding="utf-8"))
        manifest = export_topic_results(
            args.corpus_manifest,
            args.clustering_manifest,
            args.topic_manifest,
            args.reassignment_manifest,
            args.evaluation_manifest,
            args.output_dir,
            config=config,
            force=args.force,
        )
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Exported records: {manifest.records}; topics: {manifest.topics}; outliers: {manifest.outliers}")
    print(f"Manifest: {args.output_dir / 'export-manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
