"""Tests for optional RAGAS answer generation helpers (no live LLM)."""

from __future__ import annotations

import asyncio

from app.llm.types import ChatMessage, GenerationConfig, StreamChunk
from app.retrieval.models import Hit
from eval.models import GoldenCase, RagasScores
from eval.ragas_runner import generate_answers, score_with_ragas
from tests.retrieval_helpers import make_chunk


class FakeLLM:
    async def generate(self, messages: list[ChatMessage], config: GenerationConfig) -> str:
        assert messages
        assert config.max_tokens > 0
        return "Multi-head attention uses parallel heads. [1]"

    def stream_generate(self, messages: list[ChatMessage], config: GenerationConfig):
        async def _gen():
            yield StreamChunk(text="x")
            yield StreamChunk(done=True)

        return _gen()


def test_generate_answers_builds_ragas_rows() -> None:
    chunk = make_chunk("1706.03762::0", "Multi-head attention runs in parallel.")
    cases = [
        GoldenCase(
            id="mha",
            question="What is multi-head attention?",
            reference_answer="Parallel attention heads.",
            relevant_arxiv_ids=["1706.03762"],
        )
    ]

    def retrieve(question: str) -> list[Hit]:
        assert "attention" in question.lower()
        return [Hit(chunk=chunk, score=1.0, rerank_score=1.0)]

    rows = asyncio.run(generate_answers(cases, retrieve=retrieve, llm=FakeLLM()))
    assert len(rows) == 1
    assert rows[0]["answer"].startswith("Multi-head")
    assert rows[0]["contexts"][0].startswith("Multi-head")
    assert rows[0]["ground_truth"] == "Parallel attention heads."


def test_score_with_ragas_empty_rows() -> None:
    scores = score_with_ragas([])
    assert scores.skipped is True
    assert isinstance(scores, RagasScores)
