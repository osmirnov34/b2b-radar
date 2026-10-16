#!/usr/bin/env python3
"""Load one verified canonical feedback snapshot into PostgreSQL."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Sequence

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_settings
from src.operations.feedback_loading import AnalyticsLoadError, load_feedback_snapshot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Load a canonical feedback snapshot into PostgreSQL.")
    parser.add_argument("artifact", type=Path, help="Canonical feedback JSONL.")
    parser.add_argument("manifest", type=Path, help="FeedbackPreparationManifest JSON.")
    parser.add_argument("--batch-size", type=int, default=1_000)
    return parser


async def _load(artifact: Path, manifest: Path, batch_size: int) -> None:
    engine = create_async_engine(get_settings().database_url)
    try:
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        async with session_factory() as session:
            result = await load_feedback_snapshot(
                session,
                artifact,
                manifest,
                batch_size=batch_size,
            )
    finally:
        await engine.dispose()
    status = "already loaded" if result.already_loaded else "loaded"
    print(f"Status: {status}")
    print(f"Dataset ID: {result.dataset_id}")
    print(f"Records: {result.records}")
    print(f"Organizations: {result.organizations}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        asyncio.run(_load(args.artifact, args.manifest, args.batch_size))
    except (AnalyticsLoadError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
