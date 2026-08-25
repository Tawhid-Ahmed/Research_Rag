"""Sparse lexical search with BM25 (Best Matching 25).

Dense vectors miss exact keywords ("BERT", "1706.03762", rare method names).
BM25 is a TF-IDF-style ranking function that rewards query terms that are
frequent in a document but rare in the corpus. Hybrid RAG runs BM25 *and*
dense search, then fuses the lists.

``rank_bm25`` is imported lazily. Tokenization is lowercase alphanumeric
word-pieces so ``self-attention`` still matches ``attention``.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from app.ingest.models import Chunk
from app.retrieval.models import Hit

if TYPE_CHECKING:
    from rank_bm25 import BM25Okapi

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens; strips punctuation and hyphens."""

    return _TOKEN.findall(text.lower())


class BM25Index:
    """In-memory BM25 index over a list of :class:`Chunk`s."""

    def __init__(self, chunks: list[Chunk], *, bm25: BM25Okapi | None = None) -> None:
        self._chunks = list(chunks)
        self._bm25 = bm25
        self._tokenized: list[list[str]] = [tokenize(chunk.text) for chunk in self._chunks]

    @property
    def size(self) -> int:
        return len(self._chunks)

    @property
    def bm25(self) -> BM25Okapi:
        if self._bm25 is None:
            from rank_bm25 import BM25Okapi

            # Empty corpus: still construct a usable object so search returns [].
            corpus = self._tokenized or [[]]
            self._bm25 = BM25Okapi(corpus)
        return self._bm25

    def search(self, query: str, top_k: int) -> list[Hit]:
        """Return the ``top_k`` chunks with the highest BM25 scores."""

        if not self._chunks or top_k <= 0:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(
            zip(self._chunks, scores, strict=True),
            key=lambda pair: pair[1],
            reverse=True,
        )
        hits: list[Hit] = []
        for chunk, score in ranked[:top_k]:
            if float(score) <= 0.0:
                continue
            hits.append(Hit(chunk=chunk, score=float(score), bm25_score=float(score)))
        return hits
