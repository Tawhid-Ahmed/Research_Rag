"""Retrieval: hybrid BM25 + dense search and cross-encoder reranking."""

from __future__ import annotations

from app.retrieval.dense import DenseSearcher
from app.retrieval.hybrid import reciprocal_rank_fusion
from app.retrieval.models import Hit, RetrievalResult
from app.retrieval.pipeline import RetrievalPipeline, build_default_retriever
from app.retrieval.rerank import CrossEncoderReranker
from app.retrieval.sparse import BM25Index, tokenize

__all__ = [
    "BM25Index",
    "CrossEncoderReranker",
    "DenseSearcher",
    "Hit",
    "RetrievalPipeline",
    "RetrievalResult",
    "build_default_retriever",
    "reciprocal_rank_fusion",
    "tokenize",
]
