"""Run or resume the independent analytical reporting workflow."""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_settings
from src.operations.analytics_pipeline import (
    AnalyticsPipelineConfig,
    AnalyticsStage,
    preflight_analytics_pipeline,
    run_analytics_pipeline,
)


async def execute(args: argparse.Namespace) -> None:
    config = AnalyticsPipelineConfig.model_validate_json(args.config.read_text(encoding="utf-8"))
    engine = create_async_engine(get_settings().database_url)
    try:
        factory = async_sessionmaker(engine, expire_on_commit=False)
        if args.dry_run:
            print(json.dumps(await preflight_analytics_pipeline(config, factory), ensure_ascii=False, indent=2))
        else:
            state = await run_analytics_pipeline(
                config, factory, resume=args.resume,
                stop_after=AnalyticsStage(args.stop_after) if args.stop_after else None,
                progress=lambda message: print(message, flush=True),
            )
            print(json.dumps({name: state.get(name) for name in ("status", "dataset_id", "problem_run_id")}))
    finally:
        await engine.dispose()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", choices=[stage.value for stage in AnalyticsStage])
    args = parser.parse_args()
    try:
        asyncio.run(execute(args))
    except Exception:
        print("ERROR: analytics pipeline stopped. Check inputs, database access and pipeline-state.json. "
              "Raw exception details are suppressed to protect source data and credentials.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
