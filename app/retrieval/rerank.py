"""Cross-encoder reranker: score (query, chunk) pairs jointly.

A bi-encoder (our MiniLM embedder) encodes query and document *separately*
so we can pre-compute document vectors. A cross-encoder reads both together
and outputs a relevance logit — slower (can't pre-index) but much better at
fine ranking. Typical pattern: retrieve 20 cheaply, rerank the top 20, keep 5.

``sentence_transformers.CrossEncoder`` is imported lazily. The default model
``cross-encoder/ms-marco-MiniLM-L-6-v2`` is trained on MS MARCO passage
ranking and produces unbounded logits (higher = more relevant).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.retrieval.models import Hit

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """Re-score candidate hits with a query-document cross-encoder."""

    def __init__(self, model_name: str, *, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device
        self._model: CrossEncoder | None = None

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]:
        """Return ``top_k`` hits sorted by cross-encoder score (desc)."""

        if not hits or top_k <= 0:
            return []
        pairs = [(query, hit.chunk.text) for hit in hits]
        raw_scores = self.model.predict(pairs, show_progress_bar=False)
        rescored: list[Hit] = []
        for hit, raw in zip(hits, raw_scores, strict=True):
            score = float(raw)
            rescored.append(
                hit.model_copy(
                    update={
                        "score": score,
                        "rerank_score": score,
                    }
                )
            )
        rescored.sort(key=lambda item: item.score, reverse=True)
        return rescored[:top_k]
