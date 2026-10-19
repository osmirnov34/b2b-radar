"""Generate a Russian management PDF for one PostgreSQL assessment run."""

import argparse
import asyncio
import sys
from pathlib import Path

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analytics.pdf_report import write_analytical_report
from src.config import get_settings
from src.operations.analytical_report import load_analytical_report


async def generate(run_id: int, output: Path, font: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite report directory: {output}")
    engine = create_async_engine(get_settings().database_url)
    try:
        async with async_sessionmaker(engine)() as session:
            report = await load_analytical_report(session, run_id)
        write_analytical_report(report, output, font)
    finally:
        await engine.dispose()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", type=int)
    parser.add_argument("output", type=Path, help="New report directory.")
    parser.add_argument("--font", type=Path, default=Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
    args = parser.parse_args()
    try:
        asyncio.run(generate(args.run_id, args.output, args.font))
    except Exception:
        print("ERROR: report generation failed. Check run ID, database access, reporting extra, font and output path.",
              file=sys.stderr)
        return 1
    print(f"Report created: {args.output / 'report.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
