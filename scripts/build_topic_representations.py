#!/usr/bin/env python3
"""Build c-TF-IDF topic representations from fixed HDBSCAN labels."""

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

from src.ml import TopicRepresentationConfig, build_topic_representations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build topic keywords without rerunning UMAP or HDBSCAN.")
    parser.add_argument("corpus", type=Path, help="final-corpus.jsonl from stage 7.")
    parser.add_argument("labels", type=Path, help="cluster-labels.npy from stage 9.")
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--probabilities", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--clustering-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/topic-representation.example.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/topics"))
    parser.add_argument("--limit-topics", type=int, help="Represent only the first N normalized topics for a trial.")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    for label, path in (
        ("final corpus", args.corpus),
        ("final embeddings", args.embeddings),
        ("cluster labels", args.labels),
        ("cluster probabilities", args.probabilities),
        ("corpus manifest", args.corpus_manifest),
        ("clustering manifest", args.clustering_manifest),
        ("configuration", args.config),
    ):
        if not path.is_file():
            parser.error(f"{label} file not found: {path}")
    try:
        config = TopicRepresentationConfig.model_validate_json(args.config.read_text(encoding="utf-8"))
        manifest = build_topic_representations(
            args.corpus,
            args.embeddings,
            args.labels,
            args.probabilities,
            args.corpus_manifest,
            args.clustering_manifest,
            args.output_dir,
            config=config,
            force=args.force,
            limit_topics=args.limit_topics,
        )
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Records: {manifest.input_records}; topics: {manifest.topics}; outliers: {manifest.outliers}")
    print(f"Vocabulary: {manifest.quality.vocabulary_size}; empty topics: {manifest.quality.empty_topics}")
    print(f"Manifest: {args.output_dir / 'topic-representation-manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
