"""BM25 sparse retrieval tests on a fixture corpus (no network)."""

from __future__ import annotations

from app.retrieval.sparse import BM25Index, tokenize
from tests.retrieval_helpers import recall_at_k


def test_tokenize_lowercases_and_splits_hyphens() -> None:
    assert tokenize("Self-Attention!") == ["self", "attention"]


def test_bm25_ranks_attention_chunk_first(fixture_corpus) -> None:
    index = BM25Index(fixture_corpus)
    hits = index.search("self attention transformer", top_k=3)

    assert hits
    assert hits[0].chunk_id == "attn::0"
    assert hits[0].bm25_score is not None
    assert hits[0].bm25_score == hits[0].score
    assert recall_at_k(hits, {"attn::0"}, k=1) == 1.0


def test_bm25_cnn_query_does_not_surface_attention_first(fixture_corpus) -> None:
    index = BM25Index(fixture_corpus)
    hits = index.search("convolutional imagenet residual", top_k=3)

    assert hits[0].chunk_id == "cnn::0"
    assert "attn::0" not in {h.chunk_id for h in hits[:1]}


def test_bm25_empty_query_or_corpus(fixture_corpus) -> None:
    assert BM25Index([]).search("attention", top_k=5) == []
    assert BM25Index(fixture_corpus).search("   ", top_k=5) == []


def test_bm25_respects_top_k(fixture_corpus) -> None:
    hits = BM25Index(fixture_corpus).search("transformer", top_k=1)
    assert len(hits) == 1
