"""Typed models for the evaluation harness."""

from __future__ import annotations

from pydantic import BaseModel, Field


class GoldenCase(BaseModel):
    """One labeled question used for retrieval and (optionally) answer eval."""

    id: str
    question: str
    reference_answer: str
    relevant_arxiv_ids: list[str] = Field(default_factory=list)
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)


class GoldenSet(BaseModel):
    """Versioned collection of golden cases."""

    version: int = 1
    description: str = ""
    cases: list[GoldenCase]


class CaseRetrievalScores(BaseModel):
    """Per-question retrieval metrics at a fixed ``k``."""

    case_id: str
    hit: bool
    recall: float
    reciprocal_rank: float
    purity: float = 0.0
    top_ids: list[str] = Field(default_factory=list)


class AggregatedScores(BaseModel):
    """Mean metrics over a golden set."""

    hit_rate: float
    recall_at_k: float
    mrr: float
    purity_at_k: float = 0.0
    n: int
    k: int
    cases: list[CaseRetrievalScores] = Field(default_factory=list)


class AblationRow(BaseModel):
    """One retrieval configuration compared in the before/after report."""

    name: str
    description: str
    scores: AggregatedScores


class RagasScores(BaseModel):
    """Optional answer-quality metrics from RAGAS (or a test double)."""

    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    n: int = 0
    skipped: bool = False
    detail: str = ""


class EvalReport(BaseModel):
    """Full harness output written as JSON/markdown."""

    golden_path: str
    corpus_size: int
    ablations: list[AblationRow]
    ragas: RagasScores | None = None
