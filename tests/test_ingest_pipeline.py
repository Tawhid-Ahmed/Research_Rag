"""End-to-end orchestration tests using in-memory fakes for every stage.

No network, no PDFs, no model weights -- we substitute each protocol with a
trivial fake to assert the pipeline wires fetch -> parse -> chunk -> embed ->
index correctly and reports accurate statistics.
"""

from __future__ import annotations

from pathlib import Path

from app.ingest.chunk import ChunkSpan
from app.ingest.models import Chunk, Page, Paper
from app.ingest.pipeline import IngestionPipeline


class FakeFetcher:
    def __init__(self, papers: list[Paper]) -> None:
        self._papers = papers
        self.calls: list[dict[str, object]] = []

    def fetch(
        self,
        *,
        query: str | None = None,
        ids: list[str] | None = None,
        max_results: int = 10,
        download: bool = True,
    ) -> list[Paper]:
        self.calls.append({"query": query, "ids": ids, "max_results": max_results})
        return self._papers


class FakeParser:
    def __init__(self, pages_by_path: dict[str, list[Page]]) -> None:
        self._pages_by_path = pages_by_path

    def extract_pages(self, pdf_path: object) -> list[Page]:
        return self._pages_by_path.get(str(pdf_path), [])


class FakeChunker:
    def chunk(self, pages: list[Page]) -> list[ChunkSpan]:
        return [
            ChunkSpan(
                chunk_index=i,
                text=page.text,
                token_count=len(page.text.split()),
                page_start=page.page_number,
                page_end=page.page_number,
            )
            for i, page in enumerate(pages)
        ]


class FakeEmbedder:
    def __init__(self) -> None:
        self.embedded: list[str] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.embedded.extend(texts)
        return [[float(len(t)), 1.0] for t in texts]


class FakeStore:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.embeddings: list[list[float]] = []

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        self.chunks.extend(chunks)
        self.embeddings.extend(embeddings)

    def count(self) -> int:
        return len(self.chunks)


def _build(papers, pages_by_path):
    fetcher = FakeFetcher(papers)
    embedder = FakeEmbedder()
    store = FakeStore()
    pipeline = IngestionPipeline(
        fetcher=fetcher,
        parser=FakeParser(pages_by_path),
        chunker=FakeChunker(),
        embedder=embedder,
        store=store,
    )
    return pipeline, fetcher, embedder, store


def test_full_pipeline_indexes_chunks_with_citation_metadata() -> None:
    paper = Paper(
        arxiv_id="1706.03762",
        title="Attention Is All You Need",
        pdf_url="https://arxiv.org/pdf/1706.03762",
        pdf_path=Path("/tmp/1706.03762.pdf"),
    )
    pages = [Page(page_number=1, text="alpha beta"), Page(page_number=2, text="gamma")]
    pipeline, fetcher, embedder, store = _build([paper], {str(paper.pdf_path): pages})

    report = pipeline.run(query="transformers", max_results=5)

    assert report.papers_fetched == 1
    assert report.papers_indexed == 1
    assert report.pages_parsed == 2
    assert report.chunks_indexed == 2
    assert report.collection_count == 2
    assert report.skipped == []

    # Query args forwarded to the fetcher.
    assert fetcher.calls[0] == {"query": "transformers", "ids": None, "max_results": 5}

    # Chunk ids are deterministic and carry citation metadata.
    assert [c.chunk_id for c in store.chunks] == ["1706.03762::0", "1706.03762::1"]
    assert store.chunks[1].page_start == 2
    assert store.chunks[0].source_url == "https://arxiv.org/pdf/1706.03762"
    # Every chunk was embedded exactly once.
    assert embedder.embedded == ["alpha beta", "gamma"]
    assert len(store.embeddings) == 2


def test_paper_without_pdf_is_skipped() -> None:
    paper = Paper(arxiv_id="x", title="No PDF", pdf_path=None)
    pipeline, _, _, store = _build([paper], {})

    report = pipeline.run(ids=["x"])

    assert report.papers_indexed == 0
    assert report.chunks_indexed == 0
    assert store.count() == 0
    assert report.skipped == ["x: no PDF downloaded"]


def test_paper_with_no_text_is_skipped() -> None:
    paper = Paper(arxiv_id="y", title="Empty", pdf_path=Path("/tmp/y.pdf"))
    pipeline, _, _, _ = _build([paper], {str(paper.pdf_path): []})

    report = pipeline.run(ids=["y"])

    assert report.papers_indexed == 0
    assert report.skipped == ["y: no extractable text"]
