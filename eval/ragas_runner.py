"""Optional RAGAS answer-quality scoring.

Retrieval metrics always run. RAGAS needs an LLM judge and is skipped unless
explicitly enabled. Tests inject a fake scorer so CI stays offline.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from app.api.rag import build_rag_messages
from app.llm.types import ChatMessage, GenerationConfig
from app.retrieval.models import Hit
from eval.models import GoldenCase, RagasScores

logger = logging.getLogger(__name__)

RagasScorer = Callable[[list[dict[str, Any]]], RagasScores]


class AnswerGenerator(Protocol):
    async def generate(self, messages: list[ChatMessage], config: GenerationConfig) -> str: ...


async def generate_answers(
    cases: Sequence[GoldenCase],
    *,
    retrieve: Callable[[str], list[Hit]],
    llm: AnswerGenerator,
    config: GenerationConfig | None = None,
) -> list[dict[str, Any]]:
    """Retrieve context and generate an answer for each golden question."""

    gen_config = config or GenerationConfig()
    rows: list[dict[str, Any]] = []
    for case in cases:
        hits = retrieve(case.question)
        messages = build_rag_messages(case.question, hits)
        answer = await llm.generate(messages, gen_config)
        rows.append(
            {
                "question": case.question,
                "answer": answer,
                "contexts": [hit.chunk.text for hit in hits],
                "ground_truth": case.reference_answer,
                "case_id": case.id,
            }
        )
    return rows


def generate_answers_sync(
    cases: Sequence[GoldenCase],
    *,
    retrieve: Callable[[str], list[Hit]],
    llm: AnswerGenerator,
    config: GenerationConfig | None = None,
) -> list[dict[str, Any]]:
    """Sync wrapper around :func:`generate_answers` for the CLI."""

    return asyncio.run(generate_answers(cases, retrieve=retrieve, llm=llm, config=config))


def score_with_ragas(rows: list[dict[str, Any]]) -> RagasScores:
    """Run faithfulness / answer relevancy / context precision via RAGAS.

    Requires a working LLM for the judge (RAGAS defaults to OpenAI unless the
    environment is configured otherwise). Raises on import/runtime failure so
    the CLI can surface a clear skip message.
    """

    if not rows:
        return RagasScores(n=0, skipped=True, detail="no rows to score")

    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, faithfulness

    dataset = Dataset.from_list(
        [
            {
                "question": row["question"],
                "answer": row["answer"],
                "contexts": row["contexts"],
                "ground_truth": row["ground_truth"],
            }
            for row in rows
        ]
    )
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
    )
    # ragas returns a Result-like mapping; normalize to floats.
    mapping = dict(result) if hasattr(result, "items") else {}
    return RagasScores(
        faithfulness=_as_float(mapping.get("faithfulness")),
        answer_relevancy=_as_float(mapping.get("answer_relevancy")),
        context_precision=_as_float(mapping.get("context_precision")),
        n=len(rows),
        skipped=False,
        detail="ragas",
    )


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
