"""Pure helpers for turning API payloads into markdown (no Streamlit)."""

from __future__ import annotations

from typing import Any


def citation_markdown(citation: dict[str, Any], index: int) -> str:
    """Format one citation as a markdown bullet with page range and link."""

    title = str(citation.get("title") or "Untitled")
    arxiv_id = str(citation.get("arxiv_id") or "")
    page_start = citation.get("page_start", "?")
    page_end = citation.get("page_end", "?")
    score = citation.get("score")
    url = str(citation.get("source_url") or "")
    excerpt = str(citation.get("excerpt") or "").strip()
    score_bit = f" · score `{score:.3f}`" if isinstance(score, int | float) else ""
    link = f" ([PDF]({url}))" if url else ""
    header = f"**[{index}] {title}** (`{arxiv_id}`) p.{page_start}–{page_end}{score_bit}{link}"
    if excerpt:
        return f"{header}\n> {excerpt}"
    return header


def citations_markdown(citations: list[dict[str, Any]]) -> str:
    """Join citations into a markdown section."""

    if not citations:
        return "_No citations._"
    lines = [citation_markdown(item, i) for i, item in enumerate(citations, start=1)]
    return "\n\n".join(lines)
