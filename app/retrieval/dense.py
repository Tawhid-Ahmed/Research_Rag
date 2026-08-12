"""Dense vector search: embed the query, nearest-neighbor in Chroma.

Uses the same embedder as ingestion so the query vector lives in the same
space as the indexed chunks. Similarity is cosine (higher is better).
"""

from __future__ import annotations

from typing import Protocol

from app.ingest.models import Chunk
from app.retrieval.models import Hit


class Embedder(Protocol):
    def embed_query(self, text: str) -> list[float]: ...


class VectorSearcher(Protocol):
    def query_by_embedding(
        self,
        embedding: list[float],
        top_k: int,
    ) -> list[tuple[Chunk, float]]: ...


class DenseSearcher:
    """Embed a query and return the nearest stored chunks."""

    def __init__(self, store: VectorSearcher, embedder: Embedder) -> None:
        self.store = store
        self.embedder = embedder

    def search(self, query: str, top_k: int) -> list[Hit]:
        if top_k <= 0 or not query.strip():
            return []
        embedding = self.embedder.embed_query(query)
        pairs = self.store.query_by_embedding(embedding, top_k)
        return [Hit(chunk=chunk, score=score, dense_score=score) for chunk, score in pairs]
