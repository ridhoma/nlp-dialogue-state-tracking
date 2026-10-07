"""
CLI runner for the ABCD dataset ingestion pipeline.

Executes the data preparation workflow: downloads raw corpus files, parses
conversations and turns, derives intent schemas, and serializes outputs to
Parquet and SQLite.

Usage:
    uv run python scripts/run_abcd_ingestion.py [--force-download] [--no-sqlite] [--verbose]
"""

import argparse
import logging
import sys

from abcd_helper.ingestion import run_ingestion

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest raw ABCD dataset files and generate Parquet and SQLite tables."
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download of raw files from the source repository even if cached.",
    )
    parser.add_argument(
        "--no-sqlite",
        action="store_true",
        help="Skip export to SQLite database (generate Parquet files only).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    try:
        run_ingestion(
            force_download=args.force_download,
            write_sqlite=not args.no_sqlite,
        )
        return 0
    except Exception:
        logger.exception("Ingestion failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
