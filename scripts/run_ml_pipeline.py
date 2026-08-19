#!/usr/bin/env python3
"""Run, resume, publish, or roll back the offline ML pipeline."""

# ruff: noqa: E402, T201 -- standalone operational CLI configures imports and reports status.

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

from src.operations.ml_pipeline import (
    DryRunStatus,
    PipelineConfig,
    PipelineStage,
    dry_run_pipeline,
    publish_snapshot,
    render_dry_run_report,
    rollback_snapshot,
    run_pipeline,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Operate the checksum-bound offline ML pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Start or resume a pipeline run.")
    run.add_argument("config", type=Path)
    run.add_argument("--run-dir", type=Path)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--restart-from", type=PipelineStage, choices=list(PipelineStage))
    run.add_argument("--stop-after", type=PipelineStage, choices=list(PipelineStage))

    dry_run = subparsers.add_parser("dry-run", help="Validate and plan a run without writing files.")
    dry_run.add_argument("config", type=Path)
    dry_run.add_argument("--run-dir", type=Path)
    dry_run.add_argument("--resume", action="store_true")
    dry_run.add_argument("--restart-from", type=PipelineStage, choices=list(PipelineStage))
    dry_run.add_argument("--stop-after", type=PipelineStage, choices=list(PipelineStage))
    dry_run.add_argument("--verbose", action="store_true")

    publish = subparsers.add_parser("publish", help="Atomically publish a strictly passing export.")
    publish.add_argument("export_dir", type=Path)
    publish.add_argument("publish_root", type=Path)
    publish.add_argument("run_id")

    rollback = subparsers.add_parser("rollback", help="Return current to the previous valid snapshot.")
    rollback.add_argument("publish_root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command in {"run", "dry-run"}:
            if not args.config.is_file():
                parser.error(f"pipeline config not found: {args.config}")
            if args.resume and args.run_dir is None:
                parser.error("--resume requires --run-dir")
            config = PipelineConfig.model_validate_json(args.config.read_text(encoding="utf-8"))
            preflight = dry_run_pipeline(
                config,
                PROJECT_ROOT,
                config_path=args.config,
                run_dir=args.run_dir,
                resume=args.resume,
                restart_from=args.restart_from,
                stop_after=args.stop_after,
            )
            print(render_dry_run_report(preflight, verbose=getattr(args, "verbose", False)))
            if args.command == "dry-run":
                return 2 if preflight.status == DryRunStatus.BLOCKED else 0
            if not preflight.can_run:
                print("ERROR: real run refused because dry-run is blocked", file=sys.stderr)
                return 2
            result = run_pipeline(
                config,
                PROJECT_ROOT,
                run_dir=args.run_dir,
                resume=args.resume,
                restart_from=args.restart_from,
                stop_after=args.stop_after,
            )
            print(f"Run: {result.run_id}; status: {result.status.value}")
            print(f"Manifest: {(args.run_dir or config.runs_root / result.run_id) / 'run-manifest.json'}")
            if result.message:
                print(result.message)
            return 0 if result.status.value in {"completed", "awaiting_review", "partial"} else 1
        if args.command == "publish":
            manifest = publish_snapshot(args.export_dir, args.publish_root, args.run_id)
            print(f"Published manifest: {manifest}")
            return 0
        manifest = rollback_snapshot(args.publish_root)
        print(f"Rolled back manifest: {manifest}")
        return 0  # noqa: TRY300 -- command branches return their own process status.
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
