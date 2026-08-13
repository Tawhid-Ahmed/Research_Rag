"""Ablation runners: dense-only vs hybrid vs hybrid+rerank.

Each variant reuses :class:`~app.retrieval.pipeline.RetrievalPipeline` with
different searchers / rerankers so the comparison stays fair (same corpus,
same ``k``).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

from app.ingest.models import Chunk
from app.retrieval.hybrid import reciprocal_rank_fusion
from app.retrieval.models import Hit, RetrievalResult
from app.retrieval.pipeline import RetrievalPipeline
from app.retrieval.sparse import BM25Index
from eval.metrics import aggregate, score_case
from eval.models import AblationRow, AggregatedScores, GoldenCase


class Searcher(Protocol):
    def search(self, query: str, top_k: int) -> list[Hit]: ...


class RerankerLike(Protocol):
    def rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]: ...


class EmptySearcher:
    """Sparse/dense stub that contributes no candidates."""

    def search(self, query: str, top_k: int) -> list[Hit]:  # noqa: ARG002
        return []


class IdentityReranker:
    """Keep fused order; truncate to ``top_k`` (no cross-encoder)."""

    def rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]:  # noqa: ARG002
        return list(hits[:top_k])


def build_dense_only(
    dense: Searcher,
    *,
    corpus_size: int,
    retrieval_top_k: int,
    rerank_top_k: int,
) -> RetrievalPipeline:
    """Dense vector search only (no BM25, no cross-encoder)."""

    return RetrievalPipeline(
        sparse=EmptySearcher(),
        dense=dense,
        reranker=IdentityReranker(),
        corpus_size=corpus_size,
        retrieval_top_k=retrieval_top_k,
        rerank_top_k=rerank_top_k,
    )


def build_hybrid(
    sparse: Searcher,
    dense: Searcher,
    *,
    corpus_size: int,
    retrieval_top_k: int,
    rerank_top_k: int,
) -> RetrievalPipeline:
    """BM25 + dense RRF without cross-encoder reranking."""

    return RetrievalPipeline(
        sparse=sparse,
        dense=dense,
        reranker=IdentityReranker(),
        corpus_size=corpus_size,
        retrieval_top_k=retrieval_top_k,
        rerank_top_k=rerank_top_k,
    )


def build_hybrid_rerank(
    sparse: Searcher,
    dense: Searcher,
    reranker: RerankerLike,
    *,
    corpus_size: int,
    retrieval_top_k: int,
    rerank_top_k: int,
) -> RetrievalPipeline:
    """Full production stack: hybrid fusion + cross-encoder."""

    return RetrievalPipeline(
        sparse=sparse,
        dense=dense,
        reranker=reranker,
        corpus_size=corpus_size,
        retrieval_top_k=retrieval_top_k,
        rerank_top_k=rerank_top_k,
    )


def evaluate_retriever(
    cases: Sequence[GoldenCase],
    search_fn: Callable[[str], RetrievalResult],
    *,
    k: int,
) -> AggregatedScores:
    """Run ``search_fn`` on each golden question and aggregate metrics."""

    scored = [score_case(case, search_fn(case.question).hits, k=k) for case in cases]
    return aggregate(scored, k=k)


def run_ablations(
    cases: Sequence[GoldenCase],
    *,
    chunks: Sequence[Chunk],
    dense: Searcher,
    reranker: RerankerLike,
    retrieval_top_k: int,
    rerank_top_k: int,
    k: int,
) -> list[AblationRow]:
    """Compare dense-only / hybrid / hybrid+rerank on the same golden set."""

    corpus_size = len(chunks)
    sparse = BM25Index(list(chunks))

    variants: list[tuple[str, str, RetrievalPipeline]] = [
        (
            "dense_only",
            "Dense MiniLM + Chroma only (no BM25, no rerank)",
            build_dense_only(
                dense,
                corpus_size=corpus_size,
                retrieval_top_k=retrieval_top_k,
                rerank_top_k=rerank_top_k,
            ),
        ),
        (
            "hybrid",
            "BM25 + dense fused with RRF (no cross-encoder)",
            build_hybrid(
                sparse,
                dense,
                corpus_size=corpus_size,
                retrieval_top_k=retrieval_top_k,
                rerank_top_k=rerank_top_k,
            ),
        ),
        (
            "hybrid_rerank",
            "BM25 + dense RRF + cross-encoder rerank (production)",
            build_hybrid_rerank(
                sparse,
                dense,
                reranker,
                corpus_size=corpus_size,
                retrieval_top_k=retrieval_top_k,
                rerank_top_k=rerank_top_k,
            ),
        ),
    ]

    rows: list[AblationRow] = []
    for name, description, pipeline in variants:
        scores = evaluate_retriever(cases, pipeline.search, k=k)
        rows.append(AblationRow(name=name, description=description, scores=scores))
    return rows


def fuse_for_debug(sparse_hits: list[Hit], dense_hits: list[Hit], top_k: int) -> list[Hit]:
    """Expose RRF for unit tests that do not need a full pipeline."""

    return reciprocal_rank_fusion([sparse_hits, dense_hits], top_k=top_k)
