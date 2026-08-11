"""Token-aware chunking with page tracking for citations.

Why token-aware (not character- or word-based)? Embedding and LLM context
limits are measured in *tokens*, so chunking on the same unit the models use
gives predictable, dense chunks and avoids silently truncating mid-chunk.

The chunker packs the tokens of all pages into a single stream while remembering
which page each token came from. A sliding window of ``chunk_size`` tokens with
``chunk_overlap`` tokens of overlap then produces chunks that each know their
``page_start``/``page_end`` -- exactly the span needed to cite a source later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.ingest.models import Page


class TokenEncoder(Protocol):
    """Minimal reversible tokenizer interface (satisfied by ``tiktoken``)."""

    def encode(self, text: str) -> list[int]: ...

    def decode(self, tokens: list[int]) -> str: ...


@dataclass(frozen=True)
class ChunkSpan:
    """A chunk's text and the page range it was drawn from."""

    chunk_index: int
    text: str
    token_count: int
    page_start: int
    page_end: int


def get_default_encoder() -> TokenEncoder:
    """Return the ``cl100k_base`` tiktoken encoder (lazy import)."""

    import tiktoken

    return tiktoken.get_encoding("cl100k_base")


class TokenChunker:
    """Splits pages into overlapping, token-bounded :class:`ChunkSpan`s."""

    def __init__(
        self,
        *,
        chunk_size: int,
        chunk_overlap: int,
        encoder: TokenEncoder | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError("chunk_overlap must satisfy 0 <= overlap < chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._encoder = encoder

    @property
    def encoder(self) -> TokenEncoder:
        """Lazily resolve the tiktoken encoder on first use."""

        if self._encoder is None:
            self._encoder = get_default_encoder()
        return self._encoder

    def chunk(self, pages: list[Page]) -> list[ChunkSpan]:
        """Pack ``pages`` into overlapping token windows with page tracking."""

        tokens, token_pages = self._encode_with_pages(pages)
        if not tokens:
            return []

        stride = self.chunk_size - self.chunk_overlap
        spans: list[ChunkSpan] = []
        start = 0
        index = 0
        total = len(tokens)
        while start < total:
            end = min(start + self.chunk_size, total)
            window = tokens[start:end]
            text = self.encoder.decode(window).strip()
            if text:
                spans.append(
                    ChunkSpan(
                        chunk_index=index,
                        text=text,
                        token_count=len(window),
                        page_start=token_pages[start],
                        page_end=token_pages[end - 1],
                    )
                )
                index += 1
            if end == total:
                break
            start += stride
        return spans

    def _encode_with_pages(self, pages: list[Page]) -> tuple[list[int], list[int]]:
        """Flatten pages into a token stream plus a parallel page-number list."""

        tokens: list[int] = []
        token_pages: list[int] = []
        for page in pages:
            page_tokens = self.encoder.encode(page.text)
            tokens.extend(page_tokens)
            token_pages.extend([page.page_number] * len(page_tokens))
        return tokens, token_pages
