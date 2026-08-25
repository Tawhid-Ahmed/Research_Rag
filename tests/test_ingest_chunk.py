"""Tests for the token-aware chunker.

We inject a deterministic word-level encoder so chunk boundaries, overlap, and
page tracking can be asserted exactly -- without depending on tiktoken or any
network/model download.
"""

from __future__ import annotations

import pytest

from app.ingest.chunk import TokenChunker
from app.ingest.models import Page


class WordEncoder:
    """A reversible whitespace tokenizer: one token per word.

    Maintains a shared vocab so ``decode(encode(x))`` round-trips. Token ids are
    indices into ``self.vocab``; decoding joins the words with single spaces.
    """

    def __init__(self) -> None:
        self.vocab: list[str] = []
        self._index: dict[str, int] = {}

    def encode(self, text: str) -> list[int]:
        tokens: list[int] = []
        for word in text.split():
            if word not in self._index:
                self._index[word] = len(self.vocab)
                self.vocab.append(word)
            tokens.append(self._index[word])
        return tokens

    def decode(self, tokens: list[int]) -> str:
        return " ".join(self.vocab[token] for token in tokens)


def _pages(*texts: str) -> list[Page]:
    return [Page(page_number=i, text=t) for i, t in enumerate(texts, start=1)]


def test_chunk_respects_size_and_overlap() -> None:
    encoder = WordEncoder()
    text = " ".join(f"w{i}" for i in range(10))
    chunker = TokenChunker(chunk_size=4, chunk_overlap=1, encoder=encoder)

    spans = chunker.chunk(_pages(text))

    # stride = size - overlap = 3 -> windows [0:4], [3:7], [6:10] cover all 10.
    assert [s.token_count for s in spans] == [4, 4, 4]
    assert [s.chunk_index for s in spans] == [0, 1, 2]
    # Overlap: last word of chunk 0 reappears as first word of chunk 1.
    assert spans[0].text.split()[-1] == spans[1].text.split()[0]


def test_chunk_tracks_page_ranges() -> None:
    encoder = WordEncoder()
    # Page 1 has 3 tokens, page 2 has 3 tokens.
    chunker = TokenChunker(chunk_size=4, chunk_overlap=0, encoder=encoder)

    spans = chunker.chunk(_pages("a b c", "d e f"))

    assert spans[0].page_start == 1
    # Window of 4 tokens spans the page-1/page-2 boundary.
    assert spans[0].page_end == 2
    assert spans[1].page_start == 2
    assert spans[1].page_end == 2


def test_empty_pages_produce_no_chunks() -> None:
    chunker = TokenChunker(chunk_size=4, chunk_overlap=1, encoder=WordEncoder())
    assert chunker.chunk([]) == []
    assert chunker.chunk(_pages("", "")) == []


def test_invalid_overlap_rejected() -> None:
    with pytest.raises(ValueError):
        TokenChunker(chunk_size=4, chunk_overlap=4, encoder=WordEncoder())
    with pytest.raises(ValueError):
        TokenChunker(chunk_size=0, chunk_overlap=0, encoder=WordEncoder())
