"""Shared helpers for retrieval tests (imported explicitly; not a pytest plugin)."""

from __future__ import annotations

from app.ingest.models import Chunk
from app.retrieval.models import Hit


def make_chunk(
    chunk_id: str,
    text: str,
    *,
    title: str = "Untitled",
    page: int = 1,
) -> Chunk:
    arxiv_id = chunk_id.split("::", 1)[0]
    return Chunk(
        chunk_id=chunk_id,
        arxiv_id=arxiv_id,
        title=title,
        text=text,
        token_count=len(text.split()),
        chunk_index=0,
        page_start=page,
        page_end=page,
    )


def recall_at_k(hits: list[Hit], relevant: set[str], k: int) -> float:
    """Fraction of relevant chunk ids recovered in the top-k hits."""

    if not relevant or k <= 0:
        return 0.0
    retrieved = {hit.chunk_id for hit in hits[:k]}
    return len(retrieved & relevant) / len(relevant)
