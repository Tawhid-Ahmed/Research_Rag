"""Hybrid fusion: Reciprocal Rank Fusion (RRF) over ranked ID lists.

BM25 scores and cosine similarities are on incompatible scales, so we do
**not** add them. RRF ignores raw scores and only uses *rank*:

    RRF(d) = sum_i  1 / (k + rank_i(d))

``k`` (default 60) dampens the first-place spike so a doc that is #1 in one
list and missing from the other does not dominate a doc that is #3 in both.

This is the standard fusion used in many production hybrid search systems
(Elasticsearch RRF, LlamaIndex, etc.).
"""

from __future__ import annotations

from app.ingest.models import Chunk
from app.retrieval.models import Hit

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    rankings: list[list[Hit]],
    *,
    k: int = DEFAULT_RRF_K,
    top_k: int | None = None,
) -> list[Hit]:
    """Fuse ranked hit lists into one list sorted by RRF score (desc)."""

    chunks_by_id: dict[str, Chunk] = {}
    bm25_by_id: dict[str, float] = {}
    dense_by_id: dict[str, float] = {}
    rrf_by_id: dict[str, float] = {}

    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            chunk_id = hit.chunk_id
            chunks_by_id[chunk_id] = hit.chunk
            rrf_by_id[chunk_id] = rrf_by_id.get(chunk_id, 0.0) + 1.0 / (k + rank)
            if hit.bm25_score is not None:
                bm25_by_id[chunk_id] = hit.bm25_score
            if hit.dense_score is not None:
                dense_by_id[chunk_id] = hit.dense_score

    fused = sorted(rrf_by_id.items(), key=lambda item: item[1], reverse=True)
    if top_k is not None:
        fused = fused[:top_k]

    return [
        Hit(
            chunk=chunks_by_id[chunk_id],
            score=rrf_score,
            rrf_score=rrf_score,
            bm25_score=bm25_by_id.get(chunk_id),
            dense_score=dense_by_id.get(chunk_id),
        )
        for chunk_id, rrf_score in fused
    ]
