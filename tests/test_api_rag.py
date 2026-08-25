"""Unit tests for RAG citation + guardrail helpers."""

from __future__ import annotations

from app.api.rag import (
    REFUSAL_MESSAGE,
    build_rag_messages,
    confidence_from_hits,
    hits_to_citations,
    should_refuse,
)
from app.ingest.models import Chunk
from app.retrieval.models import Hit


def _hit(chunk_id: str, text: str, score: float) -> Hit:
    chunk = Chunk(
        chunk_id=chunk_id,
        arxiv_id=chunk_id.split("::", 1)[0],
        title="Paper",
        text=text,
        token_count=len(text.split()),
        chunk_index=0,
        page_start=1,
        page_end=2,
        source_url="https://arxiv.org/pdf/x",
    )
    return Hit(chunk=chunk, score=score, rerank_score=score)


def test_hits_to_citations_and_confidence() -> None:
    hits = [_hit("1706.03762::0", "attention is useful", 0.8)]
    citations = hits_to_citations(hits)
    assert citations[0].arxiv_id == "1706.03762"
    assert citations[0].page_start == 1
    assert confidence_from_hits(hits) == 0.8


def test_should_refuse_below_threshold() -> None:
    assert should_refuse(0.1, 0.2) is True
    assert should_refuse(0.5, 0.2) is False


def test_build_rag_messages_includes_numbered_context() -> None:
    hits = [_hit("a::0", "context text", 0.9)]
    messages = build_rag_messages("What is attention?", hits)
    assert messages[0].role == "system"
    assert "[1]" in messages[1].content
    assert "context text" in messages[1].content
    assert REFUSAL_MESSAGE  # imported for documentation/smoke
