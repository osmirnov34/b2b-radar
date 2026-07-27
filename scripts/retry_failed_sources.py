#!/usr/bin/env python3
"""One-off ops script: re-run document extraction for every source stuck in ingest_status=FAILED.

Typical cause: a YouTube API key expired/hit quota mid-run, failing every source processed after
that point. Fix the key set first (via /api-keys), then run this to retry the failures — it does
NOT touch sources that already succeeded or are still pending. Same action as the UI's
"Перезапустить упавшие" button; both call pipeline.reprocess_failed_sources.

Usage:
    python scripts/retry_failed_sources.py --document-limit 100
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from src.pipeline import reprocess_failed_sources

logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retry all sources with ingest_status=failed.")
    parser.add_argument("--document-limit", type=int, default=100, help="Comment limit per source (same as app.py).")
    args = parser.parse_args()
    asyncio.run(reprocess_failed_sources(args.document_limit))
