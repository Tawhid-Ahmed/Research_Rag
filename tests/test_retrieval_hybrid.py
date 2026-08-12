"""RRF fusion and dense searcher tests with in-memory fakes."""

from __future__ import annotations

from app.ingest.models import Chunk
from app.retrieval.dense import DenseSearcher
from app.retrieval.hybrid import reciprocal_rank_fusion
from app.retrieval.models import Hit
from app.retrieval.sparse import tokenize
from tests.retrieval_helpers import recall_at_k


def _hit(
    chunk: Chunk, score: float, *, bm25: float | None = None, dense: float | None = None
) -> Hit:
    return Hit(chunk=chunk, score=score, bm25_score=bm25, dense_score=dense)


def test_rrf_rewards_docs_that_appear_in_both_lists(fixture_corpus) -> None:
    attn, bert, cnn = fixture_corpus
    # attn is #2 sparse / #1 dense → should beat cnn which is only #1 sparse.
    sparse = [_hit(cnn, 9.0, bm25=9.0), _hit(attn, 4.0, bm25=4.0)]
    dense = [_hit(attn, 0.9, dense=0.9), _hit(bert, 0.2, dense=0.2)]

    fused = reciprocal_rank_fusion([sparse, dense], k=60)

    assert [h.chunk_id for h in fused][0] == "attn::0"
    attn_hit = next(h for h in fused if h.chunk_id == "attn::0")
    assert attn_hit.rrf_score == attn_hit.score
    assert attn_hit.bm25_score == 4.0
    assert attn_hit.dense_score == 0.9
    assert recall_at_k(fused, {"attn::0"}, k=1) == 1.0


def test_rrf_top_k_truncates(fixture_corpus) -> None:
    hits = [_hit(c, float(i), bm25=float(i)) for i, c in enumerate(fixture_corpus)]
    fused = reciprocal_rank_fusion([hits], k=60, top_k=2)
    assert len(fused) == 2


class BagEmbedder:
    """Tiny bag-of-words embedder so dense search is deterministic in tests."""

    vocab = (
        "attention",
        "transformer",
        "bert",
        "masked",
        "convolutional",
        "imagenet",
        "residual",
    )

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> list[float]:
        tokens = set(tokenize(text))
        raw = [1.0 if word in tokens else 0.0 for word in self.vocab]
        norm = sum(value * value for value in raw) ** 0.5 or 1.0
        return [value / norm for value in raw]


class MemoryStore:
    def __init__(self, chunks: list[Chunk], embedder: BagEmbedder) -> None:
        self._chunks = chunks
        self._vectors = embedder.embed_texts([chunk.text for chunk in chunks])

    def query_by_embedding(
        self,
        embedding: list[float],
        top_k: int,
    ) -> list[tuple[Chunk, float]]:
        scored = [
            (chunk, sum(a * b for a, b in zip(embedding, vector, strict=True)))
            for chunk, vector in zip(self._chunks, self._vectors, strict=True)
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]


def test_dense_search_ranks_by_cosine_of_bag_embeddings(fixture_corpus) -> None:
    embedder = BagEmbedder()
    searcher = DenseSearcher(MemoryStore(fixture_corpus, embedder), embedder)
    hits = searcher.search("self attention transformer", top_k=3)

    assert hits[0].chunk_id == "attn::0"
    assert hits[0].dense_score is not None
    assert recall_at_k(hits, {"attn::0"}, k=1) == 1.0


def test_dense_empty_query_returns_nothing(fixture_corpus) -> None:
    embedder = BagEmbedder()
    searcher = DenseSearcher(MemoryStore(fixture_corpus, embedder), embedder)
    assert searcher.search("   ", top_k=3) == []
