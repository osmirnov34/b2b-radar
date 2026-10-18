"""Prepare DataLens dataset queries and a dashboard assembly specification."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.operations.datalens_package import build_datalens_package


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", type=int, help="Numeric analytics.problem_runs ID.")
    parser.add_argument("output", type=Path, help="New output directory; existing files are never overwritten.")
    args = parser.parse_args()
    try:
        build_datalens_package(args.output, args.run_id)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"DataLens assembly package: {args.output}")
    print("The dashboard itself must be assembled in your DataLens workbook.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
