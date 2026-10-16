#!/usr/bin/env python3
"""Load one verified problem-assessment artifact into PostgreSQL."""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_settings  # noqa: E402
from src.operations.problem_loading import ProblemLoadError, load_problem_assessments  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Load verified problem assessments into PostgreSQL.")
    parser.add_argument("artifact", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--batch-size", type=int, default=1_000)
    parser.add_argument("--taxonomy", type=Path, help="Override taxonomy path recorded in the manifest.")
    return parser


async def _load(artifact: Path, manifest: Path, batch_size: int, taxonomy: Path | None) -> object:
    engine = create_async_engine(get_settings().database_url)
    try:
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        async with session_factory() as session:
            return await load_problem_assessments(
                session,
                artifact,
                manifest,
                batch_size=batch_size,
                taxonomy_path=taxonomy,
            )
    finally:
        await engine.dispose()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = asyncio.run(_load(args.artifact, args.manifest, args.batch_size, args.taxonomy))
    except (OSError, ProblemLoadError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Status: {'already loaded' if result.already_loaded else 'loaded'}")
    print(f"Run ID: {result.run_id}; dataset ID: {result.dataset_id}")
    print(f"Assessments: {result.assessments}")
    print(f"Categories: {result.categories}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
