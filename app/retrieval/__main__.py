"""Command-line entrypoint for hybrid retrieval.

Example::

    python -m app.retrieval --query "What is multi-head attention?"
"""

from __future__ import annotations

import argparse
import logging
import sys

from app.config import get_settings
from app.retrieval.pipeline import build_default_retriever


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.retrieval",
        description="Hybrid BM25 + dense search with cross-encoder reranking.",
    )
    parser.add_argument("--query", required=True, help="natural-language question")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    pipeline = build_default_retriever(settings)
    result = pipeline.search(args.query)

    print(f"Query: {result.query}")
    print(
        f"Corpus: {result.corpus_size} chunks | retrieved {result.retrieved} | reranked {result.reranked}"
    )
    if not result.hits:
        print("No hits. Ingest papers first: python -m app.ingest --ids 1706.03762")
        return 0

    for index, hit in enumerate(result.hits, start=1):
        chunk = hit.chunk
        print(
            f"\n[{index}] score={hit.score:.4f}  "
            f"rrf={hit.rrf_score if hit.rrf_score is not None else '-'}  "
            f"bm25={hit.bm25_score if hit.bm25_score is not None else '-'}  "
            f"dense={hit.dense_score if hit.dense_score is not None else '-'}  "
            f"rerank={hit.rerank_score if hit.rerank_score is not None else '-'}"
        )
        print(f"    {chunk.arxiv_id}  p.{chunk.page_start}-{chunk.page_end}  {chunk.title}")
        preview = chunk.text[:240].replace("\n", " ")
        print(f"    {preview}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
