"""PDF parsing: turn a downloaded PDF into cleaned, per-page text.

We deliberately split two concerns:

* :func:`clean_text` is a *pure* function (no I/O) that normalizes the messy
  text PDF extractors emit -- de-hyphenating line breaks, collapsing
  whitespace, dropping control characters. It is unit-tested directly.
* :class:`PdfParser` does the actual I/O via ``pypdf`` (imported lazily so the
  module is cheap to import and easy to stub in tests).
"""

from __future__ import annotations

import re
from pathlib import Path

from app.ingest.models import Page

# Word split across a line break: "repre-\nsentation" -> "representation".
_HYPHEN_LINEBREAK = re.compile(r"(\w)-\n(\w)")
# Any run of whitespace (incl. newlines) collapses to a single space.
_WHITESPACE = re.compile(r"\s+")
# Control characters that survive PDF extraction but carry no meaning.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_text(raw: str) -> str:
    """Normalize raw PDF text into a compact single-spaced string.

    Steps, in order: rejoin hyphenated line breaks, strip control characters,
    then collapse all remaining whitespace. The result is stable and friendly
    to token-aware chunking downstream.
    """

    if not raw:
        return ""
    text = _HYPHEN_LINEBREAK.sub(r"\1\2", raw)
    text = _CONTROL_CHARS.sub(" ", text)
    text = _WHITESPACE.sub(" ", text)
    return text.strip()


class PdfParser:
    """Extracts cleaned, per-page text from a PDF file using ``pypdf``."""

    def extract_pages(self, pdf_path: Path) -> list[Page]:
        """Return one :class:`Page` per non-empty page in ``pdf_path``.

        Pages that yield no text after cleaning (e.g. scanned figures) are
        skipped so they never become empty chunks.
        """

        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        pages: list[Page] = []
        for index, page in enumerate(reader.pages, start=1):
            cleaned = clean_text(page.extract_text() or "")
            if cleaned:
                pages.append(Page(page_number=index, text=cleaned))
        return pages
