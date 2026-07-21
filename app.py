import argparse
import asyncio
import logging

from src.pipeline import run

logging.basicConfig(level=logging.INFO)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract B2B idea sources and documents from YouTube.")
    parser.add_argument("query", help="Search query for YouTube videos.")
    parser.add_argument("--source-limit", type=int, default=50)
    parser.add_argument("--document-limit", type=int, default=100)
    args = parser.parse_args()

    asyncio.run(run(args.query, args.source_limit, args.document_limit))


if __name__ == "__main__":
    main()
