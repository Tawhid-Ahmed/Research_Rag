"""Retrieval metrics: hit-rate, recall@k, MRR.

Relevance can be labeled at **chunk** id or **arXiv paper** id level. Paper
ids are normalized (version suffix stripped) so ingest versions still match.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.retrieval.models import Hit
from eval.golden import normalize_arxiv_id
from eval.models import AggregatedScores, CaseRetrievalScores, GoldenCase


def hit_ids(hits: Sequence[Hit], *, k: int, prefer_chunk: bool) -> list[str]:
    """IDs from the top-``k`` hits used for matching against labels."""

    selected = list(hits[:k])
    if prefer_chunk:
        return [hit.chunk_id for hit in selected]
    return [normalize_arxiv_id(hit.chunk.arxiv_id) for hit in selected]


def is_hit(retrieved: Sequence[str], relevant: set[str]) -> bool:
    """True when at least one labeled relevant id appears in ``retrieved``."""

    if not relevant:
        return False
    return bool(set(retrieved) & relevant)


def recall_at_k(retrieved: Sequence[str], relevant: set[str]) -> float:
    """Fraction of labeled relevant ids recovered in the retrieved list."""

    if not relevant:
        return 0.0
    return len(set(retrieved) & relevant) / len(relevant)


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    """``1 / rank`` of the first relevant id (0 if none)."""

    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def purity_at_k(retrieved: Sequence[str], relevant: set[str]) -> float:
    """Fraction of retrieved ids that are labeled relevant."""

    if not retrieved:
        return 0.0
    return sum(1 for item in retrieved if item in relevant) / len(retrieved)


def score_case(
    case: GoldenCase,
    hits: Sequence[Hit],
    *,
    k: int,
) -> CaseRetrievalScores:
    """Score one golden case against a ranked hit list."""

    prefer_chunk = bool(case.relevant_chunk_ids)
    if prefer_chunk:
        relevant = set(case.relevant_chunk_ids)
    else:
        relevant = {normalize_arxiv_id(item) for item in case.relevant_arxiv_ids}

    retrieved = hit_ids(hits, k=k, prefer_chunk=prefer_chunk)
    return CaseRetrievalScores(
        case_id=case.id,
        hit=is_hit(retrieved, relevant),
        recall=recall_at_k(retrieved, relevant),
        reciprocal_rank=reciprocal_rank(retrieved, relevant),
        purity=purity_at_k(retrieved, relevant),
        top_ids=retrieved,
    )


def aggregate(cases: list[CaseRetrievalScores], *, k: int) -> AggregatedScores:
    """Mean hit-rate / recall@k / MRR / purity@k over scored cases."""

    n = len(cases)
    if n == 0:
        return AggregatedScores(hit_rate=0.0, recall_at_k=0.0, mrr=0.0, purity_at_k=0.0, n=0, k=k)
    return AggregatedScores(
        hit_rate=sum(1.0 for c in cases if c.hit) / n,
        recall_at_k=sum(c.recall for c in cases) / n,
        mrr=sum(c.reciprocal_rank for c in cases) / n,
        purity_at_k=sum(c.purity for c in cases) / n,
        n=n,
        k=k,
        cases=cases,
    )
