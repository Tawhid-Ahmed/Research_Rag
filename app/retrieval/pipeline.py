"""End-to-end retrieval: BM25 + dense → RRF fusion → cross-encoder rerank.

Stages are Protocol-typed so tests inject fakes (no Chroma, no model download).
:func:`build_default_retriever` loads the live Chroma collection and the
configured embedding + reranker models.
"""

from __future__ import annotations

import logging
from typing import Protocol

from app.config import Settings, get_settings
from app.ingest.models import Chunk
from app.retrieval.hybrid import DEFAULT_RRF_K, reciprocal_rank_fusion
from app.retrieval.models import Hit, RetrievalResult
from app.retrieval.sparse import BM25Index

logger = logging.getLogger(__name__)


class SparseSearcher(Protocol):
    def search(self, query: str, top_k: int) -> list[Hit]: ...


class DenseLikeSearcher(Protocol):
    def search(self, query: str, top_k: int) -> list[Hit]: ...


class Reranker(Protocol):
    def rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]: ...


class CorpusStore(Protocol):
    def get_all(self) -> list[Chunk]: ...


class RetrievalPipeline:
    """Run hybrid search then rerank; return a typed :class:`RetrievalResult`."""

    def __init__(
        self,
        *,
        sparse: SparseSearcher,
        dense: DenseLikeSearcher,
        reranker: Reranker,
        corpus_size: int,
        retrieval_top_k: int,
        rerank_top_k: int,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> None:
        if retrieval_top_k <= 0:
            raise ValueError("retrieval_top_k must be positive")
        if not 0 < rerank_top_k <= retrieval_top_k:
            raise ValueError("rerank_top_k must satisfy 0 < rerank_top_k <= retrieval_top_k")
        self.sparse = sparse
        self.dense = dense
        self.reranker = reranker
        self.corpus_size = corpus_size
        self.retrieval_top_k = retrieval_top_k
        self.rerank_top_k = rerank_top_k
        self.rrf_k = rrf_k

    def search(self, query: str) -> RetrievalResult:
        """Retrieve and rerank chunks for ``query``."""

        query = query.strip()
        if not query or self.corpus_size == 0:
            return RetrievalResult(query=query, corpus_size=self.corpus_size)

        sparse_hits = self.sparse.search(query, self.retrieval_top_k)
        dense_hits = self.dense.search(query, self.retrieval_top_k)
        fused = reciprocal_rank_fusion(
            [sparse_hits, dense_hits],
            k=self.rrf_k,
            top_k=self.retrieval_top_k,
        )
        reranked = self.reranker.rerank(query, fused, self.rerank_top_k)
        logger.info(
            "query=%r sparse=%d dense=%d fused=%d reranked=%d",
            query,
            len(sparse_hits),
            len(dense_hits),
            len(fused),
            len(reranked),
        )
        return RetrievalResult(
            query=query,
            hits=reranked,
            corpus_size=self.corpus_size,
            retrieved=len(fused),
            reranked=len(reranked),
        )


def build_default_retriever(settings: Settings | None = None) -> RetrievalPipeline:
    """Assemble the production retriever from :class:`~app.config.Settings`."""

    settings = settings or get_settings()

    from app.ingest.embed import SentenceTransformerEmbedder
    from app.ingest.store import ChromaVectorStore
    from app.retrieval.dense import DenseSearcher
    from app.retrieval.rerank import CrossEncoderReranker

    store = ChromaVectorStore(settings.chroma_persist_dir, settings.collection_name)
    chunks = store.get_all()
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    return RetrievalPipeline(
        sparse=BM25Index(chunks),
        dense=DenseSearcher(store, embedder),
        reranker=CrossEncoderReranker(settings.reranker_model),
        corpus_size=len(chunks),
        retrieval_top_k=settings.retrieval_top_k,
        rerank_top_k=settings.rerank_top_k,
    )
