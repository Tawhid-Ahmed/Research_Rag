"""Tests for ingestion data models and metadata flattening."""

from __future__ import annotations

from app.ingest.models import (
    Chunk,
    Paper,
    authors_to_str,
    categories_to_str,
)


def test_chunk_to_metadata_is_primitive_only() -> None:
    chunk = Chunk(
        chunk_id="1706.03762::0",
        arxiv_id="1706.03762",
        title="Attention Is All You Need",
        text="some passage",
        token_count=42,
        chunk_index=0,
        page_start=1,
        page_end=2,
        source_url="https://arxiv.org/pdf/1706.03762",
    )

    metadata = chunk.to_metadata()

    assert set(metadata) == {
        "arxiv_id",
        "title",
        "chunk_index",
        "token_count",
        "page_start",
        "page_end",
        "source_url",
    }
    assert all(isinstance(v, str | int) for v in metadata.values())
    assert metadata["chunk_index"] == 0
    assert metadata["page_end"] == 2


def test_paper_defaults() -> None:
    paper = Paper(arxiv_id="2005.11401", title="RAG")
    assert paper.authors == []
    assert paper.categories == []
    assert paper.pdf_path is None


def test_list_flatteners() -> None:
    assert authors_to_str(["A", "B"]) == "A; B"
    assert categories_to_str(["cs.CL", "cs.LG"]) == "cs.CL, cs.LG"
