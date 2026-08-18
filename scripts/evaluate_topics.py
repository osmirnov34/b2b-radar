#!/usr/bin/env python3
"""Evaluate fixed topic-analysis artifacts and prepare a local review sample."""

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

from src.ml import EvaluationConfig, evaluate_topics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate topic geometry, stability, quality, and human annotations.")
    parser.add_argument("embeddings", type=Path)
    parser.add_argument("source_labels", type=Path)
    parser.add_argument("final_labels", type=Path)
    parser.add_argument("final_confidence", type=Path)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--clustering-manifest", type=Path, required=True)
    parser.add_argument("--topic-manifest", type=Path, required=True)
    parser.add_argument("--reassignment-manifest", type=Path, required=True)
    parser.add_argument("--manual-annotations", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/evaluation.example.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/evaluation"))
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    paths = {
        "embeddings": args.embeddings,
        "source labels": args.source_labels,
        "final labels": args.final_labels,
        "final confidence": args.final_confidence,
        "corpus": args.corpus,
        "corpus manifest": args.corpus_manifest,
        "clustering manifest": args.clustering_manifest,
        "topic manifest": args.topic_manifest,
        "reassignment manifest": args.reassignment_manifest,
        "configuration": args.config,
    }
    if args.manual_annotations is not None:
        paths["manual annotations"] = args.manual_annotations
    for label, path in paths.items():
        if not path.is_file():
            parser.error(f"{label} file not found: {path}")
    try:
        config = EvaluationConfig.model_validate_json(args.config.read_text(encoding="utf-8"))
        manifest = evaluate_topics(
            args.embeddings,
            args.corpus,
            args.source_labels,
            args.final_labels,
            args.final_confidence,
            args.corpus_manifest,
            args.clustering_manifest,
            args.topic_manifest,
            args.reassignment_manifest,
            args.output_dir,
            config=config,
            manual_annotations_path=args.manual_annotations,
            force=args.force,
        )
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Status: {manifest.warnings or 'no warnings'}")
    print(f"Manifest: {args.output_dir / 'evaluation-manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
