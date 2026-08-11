"""Command-line entrypoint for the ingestion pipeline.

Examples
--------
Ingest the 5 most relevant papers for a search query::

    python -m app.ingest --query "retrieval augmented generation" --max-results 5

Ingest specific papers by arXiv id::

    python -m app.ingest --ids 1706.03762 2005.11401
"""

from __future__ import annotations

import argparse
import logging
import sys

from app.config import get_settings
from app.ingest.pipeline import build_default_pipeline


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.ingest",
        description="Ingest arXiv papers into the vector store.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--query", help="free-text arXiv search query")
    source.add_argument("--ids", nargs="+", help="explicit arXiv id(s)")
    parser.add_argument(
        "--max-results",
        type=int,
        default=10,
        help="maximum number of papers to ingest (default: 10)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse args, run the pipeline, and print a short summary report."""

    args = _build_arg_parser().parse_args(argv)

    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    pipeline = build_default_pipeline(settings)
    report = pipeline.run(query=args.query, ids=args.ids, max_results=args.max_results)

    print("Ingestion complete:")
    print(f"  papers fetched : {report.papers_fetched}")
    print(f"  papers indexed : {report.papers_indexed}")
    print(f"  pages parsed   : {report.pages_parsed}")
    print(f"  chunks indexed : {report.chunks_indexed}")
    print(f"  collection size: {report.collection_count}")
    if report.skipped:
        print("  skipped:")
        for item in report.skipped:
            print(f"    - {item}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
