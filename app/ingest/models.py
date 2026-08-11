"""Typed data models that flow through the ingestion pipeline.

The pipeline transforms data in stages, each with its own model:

``Paper``  -> downloaded arXiv metadata + local PDF path (fetch stage)
``Page``   -> cleaned text for one PDF page (parse stage)
``Chunk``  -> a token-bounded slice of a paper, ready to embed/index
``IngestionReport`` -> summary statistics returned to the caller/CLI

Keeping these explicit (rather than passing dicts around) gives us type
safety, validation, and a single place to define how a chunk maps to the flat
metadata Chroma can store.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

# Chroma metadata values must be primitives; list fields are flattened to
# delimited strings using these separators when we build a metadata dict.
_AUTHOR_SEP = "; "
_CATEGORY_SEP = ", "


class Paper(BaseModel):
    """An arXiv paper's metadata plus the location of its downloaded PDF."""

    arxiv_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    abstract: str = ""
    categories: list[str] = Field(default_factory=list)
    published: str = ""
    pdf_url: str = ""
    pdf_path: Path | None = None


class Page(BaseModel):
    """Cleaned text extracted from a single (1-indexed) PDF page."""

    page_number: int
    text: str


class Chunk(BaseModel):
    """A token-bounded passage of a paper, carrying citation metadata.

    ``chunk_id`` is deterministic (``{arxiv_id}::{chunk_index}``) so re-ingesting
    the same paper upserts rather than duplicates rows in the vector store.
    """

    chunk_id: str
    arxiv_id: str
    title: str
    text: str
    token_count: int
    chunk_index: int
    page_start: int
    page_end: int
    source_url: str = ""

    def to_metadata(self) -> dict[str, str | int]:
        """Flatten to Chroma-compatible metadata (primitives only)."""

        return {
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "chunk_index": self.chunk_index,
            "token_count": self.token_count,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "source_url": self.source_url,
        }


class IngestionReport(BaseModel):
    """Summary of one ingestion run, returned to the CLI/API for feedback."""

    papers_fetched: int = 0
    papers_indexed: int = 0
    pages_parsed: int = 0
    chunks_indexed: int = 0
    skipped: list[str] = Field(default_factory=list)
    collection_count: int = 0


def authors_to_str(authors: list[str]) -> str:
    """Join author names into a single Chroma-storable string."""

    return _AUTHOR_SEP.join(authors)


def categories_to_str(categories: list[str]) -> str:
    """Join arXiv categories into a single Chroma-storable string."""

    return _CATEGORY_SEP.join(categories)
