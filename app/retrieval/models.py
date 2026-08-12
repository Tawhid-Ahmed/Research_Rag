"""Typed hits returned by sparse, dense, hybrid, and rerank stages.

Each stage attaches the score it computed and leaves the others ``None``. The
final :class:`RetrievalResult` is what the API/CLI will show (and later feed
to the LLM as cited context).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.ingest.models import Chunk


class Hit(BaseModel):
    """One retrieved chunk plus the scores produced along the pipeline."""

    chunk: Chunk
    score: float
    bm25_score: float | None = None
    dense_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id


class RetrievalResult(BaseModel):
    """Ranked hits for a single query, plus the corpus size searched."""

    query: str
    hits: list[Hit] = Field(default_factory=list)
    corpus_size: int = 0
    retrieved: int = 0
    reranked: int = 0
