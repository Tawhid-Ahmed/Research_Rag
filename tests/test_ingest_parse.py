"""Tests for PDF text cleaning and page extraction.

``clean_text`` is pure and tested directly. ``PdfParser.extract_pages`` is tested
by monkeypatching ``pypdf.PdfReader`` with a fake, so no real PDF is needed.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

from app.ingest.parse import PdfParser, clean_text


def test_clean_text_rejoins_hyphenated_linebreaks() -> None:
    assert clean_text("repre-\nsentation") == "representation"


def test_clean_text_collapses_whitespace() -> None:
    assert clean_text("hello   world\n\n  again\t!") == "hello world again !"


def test_clean_text_strips_control_chars() -> None:
    assert clean_text("a\x00b\x07c") == "a b c"


def test_clean_text_empty() -> None:
    assert clean_text("") == ""
    assert clean_text("   \n\t ") == ""


class _FakePage:
    def __init__(self, text: str) -> None:
        self._text = text

    def extract_text(self) -> str:
        return self._text


class _FakeReader:
    def __init__(self, path: str) -> None:  # noqa: ARG002 - signature parity
        self.pages = [
            _FakePage("First page  text"),
            _FakePage("   "),  # blank page -> skipped
            _FakePage("Third page\ntext"),
        ]


def test_extract_pages_skips_blank_and_numbers_pages(monkeypatch) -> None:
    # ``pypdf`` may not be installed in every test env; inject a fake module so
    # the parser's lazy ``from pypdf import PdfReader`` resolves to our stub.
    fake_pypdf = types.ModuleType("pypdf")
    fake_pypdf.PdfReader = _FakeReader  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pypdf", fake_pypdf)

    pages = PdfParser().extract_pages(Path("dummy.pdf"))

    assert [p.page_number for p in pages] == [1, 3]
    assert pages[0].text == "First page text"
    assert pages[1].text == "Third page text"
