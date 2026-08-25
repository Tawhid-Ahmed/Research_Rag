"""CLI: run retrieval ablations (+ optional RAGAS) and write a report.

Examples::

    python -m eval
    python -m eval --k 5 --report eval/reports/latest.md
    python -m eval --with-ragas
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from eval.ablation import run_ablations
from eval.golden import DEFAULT_GOLDEN_PATH, load_golden_set
from eval.models import EvalReport, RagasScores
from eval.ragas_runner import generate_answers_sync, score_with_ragas
from eval.report import render_markdown, write_report

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate retrieval (+ optional RAGAS).")
    parser.add_argument(
        "--golden",
        type=Path,
        default=DEFAULT_GOLDEN_PATH,
        help="Path to golden.json",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Cutoff for hit-rate / recall@k / MRR (default: 5)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("eval/reports/latest.md"),
        help="Markdown report path (JSON sidecar written alongside)",
    )
    parser.add_argument(
        "--with-ragas",
        action="store_true",
        help="Generate answers with the configured LLM and score via RAGAS",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Debug logging",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    golden = load_golden_set(args.golden)
    if not golden.cases:
        logger.error("golden set is empty: %s", args.golden)
        return 1

    from app.config import get_settings
    from app.ingest.embed import SentenceTransformerEmbedder
    from app.ingest.store import ChromaVectorStore
    from app.retrieval.dense import DenseSearcher
    from app.retrieval.rerank import CrossEncoderReranker

    settings = get_settings()
    store = ChromaVectorStore(settings.chroma_persist_dir, settings.collection_name)
    chunks = store.get_all()
    if not chunks:
        logger.error(
            "Chroma collection is empty. Ingest first, e.g. "
            "`python -m app.ingest --ids 1706.03762 --max-results 1`"
        )
        return 1

    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    dense = DenseSearcher(store, embedder)
    reranker = CrossEncoderReranker(settings.reranker_model)

    ablations = run_ablations(
        golden.cases,
        chunks=chunks,
        dense=dense,
        reranker=reranker,
        retrieval_top_k=settings.retrieval_top_k,
        rerank_top_k=settings.rerank_top_k,
        k=args.k,
    )

    ragas_scores: RagasScores | None = None
    if args.with_ragas:
        from app.llm.factory import build_provider
        from app.llm.types import GenerationConfig
        from app.retrieval.pipeline import build_default_retriever

        retriever = build_default_retriever(settings)
        llm = build_provider(settings)
        try:
            rows = generate_answers_sync(
                golden.cases,
                retrieve=lambda q: retriever.search(q).hits,
                llm=llm,
                config=GenerationConfig(
                    temperature=settings.llm_temperature,
                    max_tokens=settings.llm_max_tokens,
                ),
            )
            ragas_scores = score_with_ragas(rows)
        except Exception as exc:  # noqa: BLE001 - CLI should not crash the ablation
            logger.exception("RAGAS failed")
            ragas_scores = RagasScores(skipped=True, detail=str(exc), n=0)

    report = EvalReport(
        golden_path=str(args.golden),
        corpus_size=len(chunks),
        ablations=ablations,
        ragas=ragas_scores,
    )
    write_report(report, args.report)
    print(render_markdown(report))
    print(f"\nWrote {args.report} and {args.report.with_suffix('.json')}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
