"""Fetch papers from arXiv: metadata + PDF download.

Wraps the ``arxiv`` client (imported lazily) behind a small, typed surface so
the rest of the pipeline depends on our :class:`Paper` model rather than the
library's result objects. Supports two query modes -- a free-text search or an
explicit list of arXiv IDs -- and caches PDFs on disk so re-runs don't re-download.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.ingest.models import Paper


def _result_to_paper(result: Any) -> Paper:
    """Map an ``arxiv.Result`` onto our :class:`Paper` model."""

    published = result.published.date().isoformat() if result.published else ""
    return Paper(
        arxiv_id=result.get_short_id(),
        title=result.title.strip(),
        authors=[author.name for author in result.authors],
        abstract=(result.summary or "").strip(),
        categories=list(result.categories),
        published=published,
        pdf_url=result.pdf_url or "",
    )


class ArxivFetcher:
    """Searches arXiv and downloads PDFs into a local cache directory."""

    def __init__(self, pdf_cache_dir: Path, *, page_size: int = 50) -> None:
        self.pdf_cache_dir = pdf_cache_dir
        self.page_size = page_size

    def fetch(
        self,
        *,
        query: str | None = None,
        ids: list[str] | None = None,
        max_results: int = 10,
        download: bool = True,
    ) -> list[Paper]:
        """Return papers matching ``query`` or ``ids`` (one is required).

        When ``download`` is true (the default) each paper's PDF is fetched into
        the cache and its local path recorded on :attr:`Paper.pdf_path`.
        """

        if not query and not ids:
            raise ValueError("provide either a search query or a list of arXiv ids")

        import arxiv

        client = arxiv.Client(page_size=self.page_size)
        search = arxiv.Search(
            query=query or "",
            id_list=ids or [],
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        papers: list[Paper] = []
        for result in client.results(search):
            paper = _result_to_paper(result)
            if download:
                paper.pdf_path = self._download(result, paper.arxiv_id)
            papers.append(paper)
        return papers

    def _download(self, result: Any, arxiv_id: str) -> Path:
        """Download a result's PDF to the cache, skipping if already present."""

        self.pdf_cache_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{arxiv_id.replace('/', '_')}.pdf"
        target = self.pdf_cache_dir / filename
        if target.exists():
            return target
        result.download_pdf(dirpath=str(self.pdf_cache_dir), filename=filename)
        return target
