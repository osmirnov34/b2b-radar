"""Export reconciled DataLens CSV files without creating cloud resources."""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_settings
from src.operations.analytics_demo import load_demo_environment
from src.operations.datalens_csv import export_datalens_csv


async def execute(args: argparse.Namespace) -> None:
    if args.demo_directory:
        load_demo_environment(args.demo_directory)
    engine = create_async_engine(get_settings().database_url)
    try:
        async with async_sessionmaker(engine)() as session:
            result = await export_datalens_csv(session, args.report_directory, args.output_directory)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        await engine.dispose()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--demo-directory", type=Path, help="Explicit local demo environment; otherwise use POSTGRES_*.")
    args = parser.parse_args()
    try:
        asyncio.run(execute(args))
    except Exception:
        print("ERROR: CSV export failed. Check report integrity, database access and aggregate consistency. "
              "Raw error details are suppressed to protect source data and credentials.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
