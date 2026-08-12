"""Pipeline orchestration + reranker order tests with fakes."""

from __future__ import annotations

import pytest

from app.retrieval.models import Hit
from app.retrieval.pipeline import RetrievalPipeline
from app.retrieval.sparse import tokenize
from tests.retrieval_helpers import recall_at_k


class FakeSearcher:
    def __init__(self, hits: list[Hit]) -> None:
        self._hits = hits
        self.queries: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int) -> list[Hit]:
        self.queries.append((query, top_k))
        return self._hits[:top_k]


class KeywordReranker:
    """Score = query-term overlap; used to prove rerank can change order."""

    def rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]:
        query_terms = set(tokenize(query))
        rescored: list[Hit] = []
        for hit in hits:
            overlap = float(len(query_terms & set(tokenize(hit.chunk.text))))
            rescored.append(hit.model_copy(update={"score": overlap, "rerank_score": overlap}))
        rescored.sort(key=lambda item: item.score, reverse=True)
        return rescored[:top_k]


def test_pipeline_fuses_then_reranks_and_reports_counts(fixture_corpus) -> None:
    attn, bert, cnn = fixture_corpus
    sparse = FakeSearcher(
        [
            Hit(chunk=cnn, score=10.0, bm25_score=10.0),
            Hit(chunk=attn, score=3.0, bm25_score=3.0),
        ]
    )
    dense = FakeSearcher(
        [
            Hit(chunk=attn, score=0.9, dense_score=0.9),
            Hit(chunk=bert, score=0.4, dense_score=0.4),
        ]
    )
    pipeline = RetrievalPipeline(
        sparse=sparse,
        dense=dense,
        reranker=KeywordReranker(),
        corpus_size=3,
        retrieval_top_k=3,
        rerank_top_k=2,
    )

    result = pipeline.search("self attention transformer")

    assert result.corpus_size == 3
    assert result.retrieved >= 1
    assert result.reranked == len(result.hits) == 2
    # Reranker should promote the attention chunk over the CNN chunk.
    assert result.hits[0].chunk_id == "attn::0"
    assert result.hits[0].rerank_score is not None
    assert recall_at_k(result.hits, {"attn::0"}, k=1) == 1.0
    assert sparse.queries[0] == ("self attention transformer", 3)


def test_reranker_changes_order_vs_rrf_alone(fixture_corpus) -> None:
    attn, bert, cnn = fixture_corpus
    # CNN is #1 in both lists → wins RRF, but has zero overlap with the query.
    sparse = FakeSearcher(
        [Hit(chunk=cnn, score=5.0, bm25_score=5.0), Hit(chunk=attn, score=1.0, bm25_score=1.0)]
    )
    dense = FakeSearcher(
        [Hit(chunk=cnn, score=0.8, dense_score=0.8), Hit(chunk=bert, score=0.1, dense_score=0.1)]
    )
    pipeline = RetrievalPipeline(
        sparse=sparse,
        dense=dense,
        reranker=KeywordReranker(),
        corpus_size=3,
        retrieval_top_k=3,
        rerank_top_k=3,
    )

    result = pipeline.search("self attention")

    assert result.hits[0].chunk_id == "attn::0"
    assert result.hits[0].chunk_id != "cnn::0"


def test_empty_query_or_corpus_short_circuits(fixture_corpus) -> None:
    pipeline = RetrievalPipeline(
        sparse=FakeSearcher([]),
        dense=FakeSearcher([]),
        reranker=KeywordReranker(),
        corpus_size=0,
        retrieval_top_k=5,
        rerank_top_k=3,
    )
    empty_corpus = pipeline.search("anything")
    assert empty_corpus.hits == []
    assert empty_corpus.corpus_size == 0

    filled = RetrievalPipeline(
        sparse=FakeSearcher([]),
        dense=FakeSearcher([]),
        reranker=KeywordReranker(),
        corpus_size=3,
        retrieval_top_k=5,
        rerank_top_k=3,
    )
    assert filled.search("   ").hits == []


def test_invalid_k_rejected() -> None:
    with pytest.raises(ValueError):
        RetrievalPipeline(
            sparse=FakeSearcher([]),
            dense=FakeSearcher([]),
            reranker=KeywordReranker(),
            corpus_size=1,
            retrieval_top_k=2,
            rerank_top_k=5,
        )
