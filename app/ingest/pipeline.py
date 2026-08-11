"""End-to-end ingestion orchestration.

Wires the five stages together:

    fetch -> parse -> chunk -> embed -> index

Each stage is referenced through a narrow :class:`typing.Protocol`, so the
pipeline depends on *behaviour*, not concrete classes. That makes the flow
trivial to unit-test with in-memory fakes (no network, no model downloads) and
keeps the heavy implementations (arxiv/pypdf/sentence-transformers/chromadb)
out of the orchestration logic.

:func:`build_default_pipeline` assembles the real, config-driven implementations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

from app.config import Settings, get_settings
from app.ingest.chunk import ChunkSpan, TokenChunker
from app.ingest.models import Chunk, IngestionReport, Page, Paper

logger = logging.getLogger(__name__)


class Fetcher(Protocol):
    def fetch(
        self,
        *,
        query: str | None = None,
        ids: list[str] | None = None,
        max_results: int = 10,
        download: bool = True,
    ) -> list[Paper]: ...


class Parser(Protocol):
    def extract_pages(self, pdf_path: Path) -> list[Page]: ...


class Chunker(Protocol):
    def chunk(self, pages: list[Page]) -> list[ChunkSpan]: ...


class Embedder(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...


class VectorStore(Protocol):
    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None: ...

    def count(self) -> int: ...


class IngestionPipeline:
    """Runs the fetch -> parse -> chunk -> embed -> index flow."""

    def __init__(
        self,
        *,
        fetcher: Fetcher,
        parser: Parser,
        chunker: Chunker,
        embedder: Embedder,
        store: VectorStore,
    ) -> None:
        self.fetcher = fetcher
        self.parser = parser
        self.chunker = chunker
        self.embedder = embedder
        self.store = store

    def run(
        self,
        *,
        query: str | None = None,
        ids: list[str] | None = None,
        max_results: int = 10,
    ) -> IngestionReport:
        """Ingest papers matching ``query``/``ids`` and return run statistics."""

        papers = self.fetcher.fetch(query=query, ids=ids, max_results=max_results)
        report = IngestionReport(papers_fetched=len(papers))

        for paper in papers:
            chunks = self._process_paper(paper, report)
            if not chunks:
                continue
            embeddings = self.embedder.embed_texts([chunk.text for chunk in chunks])
            self.store.add(chunks, embeddings)
            report.papers_indexed += 1
            report.chunks_indexed += len(chunks)
            logger.info("Indexed %s (%d chunks)", paper.arxiv_id, len(chunks))

        report.collection_count = self.store.count()
        return report

    def _process_paper(self, paper: Paper, report: IngestionReport) -> list[Chunk]:
        """Parse + chunk one paper, recording skips on the report."""

        if paper.pdf_path is None:
            report.skipped.append(f"{paper.arxiv_id}: no PDF downloaded")
            return []

        pages = self.parser.extract_pages(paper.pdf_path)
        if not pages:
            report.skipped.append(f"{paper.arxiv_id}: no extractable text")
            return []
        report.pages_parsed += len(pages)

        spans = self.chunker.chunk(pages)
        if not spans:
            report.skipped.append(f"{paper.arxiv_id}: produced no chunks")
            return []

        return [self._span_to_chunk(paper, span) for span in spans]

    @staticmethod
    def _span_to_chunk(paper: Paper, span: ChunkSpan) -> Chunk:
        """Attach paper-level citation metadata to a raw chunk span."""

        return Chunk(
            chunk_id=f"{paper.arxiv_id}::{span.chunk_index}",
            arxiv_id=paper.arxiv_id,
            title=paper.title,
            text=span.text,
            token_count=span.token_count,
            chunk_index=span.chunk_index,
            page_start=span.page_start,
            page_end=span.page_end,
            source_url=paper.pdf_url,
        )


def build_default_pipeline(settings: Settings | None = None) -> IngestionPipeline:
    """Assemble the production pipeline from application settings."""

    settings = settings or get_settings()

    # Imported here (not at module top) to keep heavy deps out of import paths.
    from app.ingest.embed import SentenceTransformerEmbedder
    from app.ingest.fetch import ArxivFetcher
    from app.ingest.parse import PdfParser
    from app.ingest.store import ChromaVectorStore

    return IngestionPipeline(
        fetcher=ArxivFetcher(settings.pdf_cache_dir),
        parser=PdfParser(),
        chunker=TokenChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        ),
        embedder=SentenceTransformerEmbedder(settings.embedding_model),
        store=ChromaVectorStore(settings.chroma_persist_dir, settings.collection_name),
    )
