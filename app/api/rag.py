"""RAG helpers: prompt construction, citations, and confidence guardrail."""

from __future__ import annotations

import math

from app.api.schemas import Citation
from app.llm.types import ChatMessage
from app.retrieval.models import Hit, RetrievalResult

REFUSAL_MESSAGE = (
    "I could not find sufficiently relevant passages in the indexed papers "
    "to answer confidently. Try rephrasing the question or ingesting more papers."
)


def _sigmoid(value: float) -> float:
    """Map unbounded cross-encoder logits into ``(0, 1)`` for confidence."""

    # Stable sigmoid for large |x|.
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_x = math.exp(value)
    return exp_x / (1.0 + exp_x)


def hits_to_citations(hits: list[Hit], *, excerpt_chars: int = 240) -> list[Citation]:
    """Map retrieval hits to API citation objects."""

    citations: list[Citation] = []
    for hit in hits:
        chunk = hit.chunk
        score = hit.rerank_score if hit.rerank_score is not None else hit.score
        excerpt = chunk.text[:excerpt_chars].replace("\n", " ").strip()
        citations.append(
            Citation(
                chunk_id=chunk.chunk_id,
                arxiv_id=chunk.arxiv_id,
                title=chunk.title,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                score=float(score),
                source_url=chunk.source_url,
                excerpt=excerpt,
            )
        )
    return citations


def confidence_from_hits(hits: list[Hit]) -> float:
    """Best-hit confidence in ``[0, 1]`` for the refuse guardrail.

    ``cross-encoder/ms-marco-MiniLM-L-6-v2`` returns raw logits (often negative
    even for on-topic chunks). Comparing those logits to a 0–1 threshold caused
    false refusals; we sigmoid the score so ``min_confidence_score`` stays a
    probability floor.
    """

    if not hits:
        return 0.0
    best = hits[0]
    score = best.rerank_score if best.rerank_score is not None else best.score
    return _sigmoid(float(score))


def should_refuse(confidence: float, min_confidence: float) -> bool:
    """Return True when retrieval confidence is below the configured floor."""

    return confidence < min_confidence


def build_rag_messages(question: str, hits: list[Hit]) -> list[ChatMessage]:
    """Build a grounded chat prompt with numbered context blocks."""

    blocks: list[str] = []
    for index, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        blocks.append(
            f"[{index}] arxiv_id={chunk.arxiv_id} "
            f"pages={chunk.page_start}-{chunk.page_end} title={chunk.title}\n"
            f"{chunk.text}"
        )
    context = "\n\n".join(blocks) if blocks else "(no context)"
    system = (
        "You are a research assistant for arXiv AI/ML papers. "
        "Answer using ONLY the provided context. "
        "Cite sources inline like [1], [2] matching the context numbers. "
        "If the context is insufficient, say you do not know."
    )
    user = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    return [
        ChatMessage(role="system", content=system),
        ChatMessage(role="user", content=user),
    ]


def summarize_retrieval(result: RetrievalResult) -> dict[str, int]:
    """Small debug payload useful in streaming metadata events."""

    return {
        "corpus_size": result.corpus_size,
        "retrieved": result.retrieved,
        "reranked": result.reranked,
    }
