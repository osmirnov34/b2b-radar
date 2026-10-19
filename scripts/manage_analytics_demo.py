"""Manage the isolated loopback PostgreSQL analytics demo; no cloud resources are created."""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_settings
from src.operations.analytics_demo import initialize_analytics_demo, inspect_analytics_demo, load_demo_environment
from src.operations.analytics_pipeline import preflight_analytics_pipeline, run_analytics_pipeline


async def execute(action: str, directory: Path) -> None:
    config = load_demo_environment(directory)
    engine = create_async_engine(get_settings().database_url)
    try:
        factory = async_sessionmaker(engine, expire_on_commit=False)
        if action == "check":
            result = await preflight_analytics_pipeline(config, factory)
            print(json.dumps({"mode": result["mode"], "preflight": "passed"}))
        elif action in {"run", "resume"}:
            state = await run_analytics_pipeline(
                config, factory, resume=action == "resume", progress=lambda message: print(message, flush=True)
            )
            print(json.dumps({key: state.get(key) for key in ("status", "dataset_id", "problem_run_id")}))
        elif action == "bi-role":
            sql = Path("scripts/datalens_reader_role.sql").read_text()
            statements = "\n".join(line for line in sql.splitlines() if not line.lstrip().startswith("--"))
            async with engine.begin() as connection:
                for statement in statements.split(";"):
                    if statement.strip():
                        await connection.exec_driver_sql(statement)
            print("Restricted BI reader role created")
        elif action == "accept":
            result = await inspect_analytics_demo(config, factory)
            path = directory / "acceptance.json"
            with path.open("x", encoding="utf-8") as output:
                output.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
            print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        await engine.dispose()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["init", "migrate", "check", "run", "resume", "bi-role", "accept", "serve"])
    parser.add_argument("--directory", type=Path, default=Path("data/demo/analytics-local"))
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    try:
        if args.action == "init":
            if args.source is None:
                parser.error("init requires --source")
            initialize_analytics_demo(args.source, args.directory)
            print(f"Private demo configuration created in {args.directory}")
        elif args.action == "migrate":
            load_demo_environment(args.directory)
            command.upgrade(Config("alembic.ini"), "head")
        elif args.action == "serve":
            load_demo_environment(args.directory)
            import uvicorn
            uvicorn.run("src.web.analytics_app:app", host="127.0.0.1", port=8001)
        else:
            asyncio.run(execute(args.action, args.directory))
    except Exception:
        print("ERROR: local demo action failed; check configuration, container health and workflow state. "
              "Exception details are suppressed to protect credentials and source reviews.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
