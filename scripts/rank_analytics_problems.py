"""Print explainable priorities for one explicit analytical problem run."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_settings
from src.operations.problem_priorities import load_problem_priorities


async def main(run_id: int) -> None:
    engine = create_async_engine(get_settings().database_url)
    try:
        async with async_sessionmaker(engine)() as session:
            priorities = await load_problem_priorities(session, run_id)
            print(json.dumps([row.model_dump(mode="json") for row in priorities], ensure_ascii=False, indent=2))
    finally:
        await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", type=int)
    asyncio.run(main(parser.parse_args().run_id))
