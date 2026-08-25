"""Eval tab: last-query stats + latest harness report if present."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from ui.client import ApiClient

_REPORT_MD = Path(__file__).resolve().parents[2] / "eval" / "reports" / "latest.md"
_REPORT_JSON = _REPORT_MD.with_suffix(".json")


def render_eval(_client: ApiClient) -> None:
    """Show last-query retrieval numbers and the offline eval report."""

    st.subheader("Eval")
    st.caption(
        "Live stats come from the last Chat query. "
        "The harness report is produced by `python -m eval` "
        "(retrieval ablation; optional `--with-ragas`)."
    )

    meta: dict[str, Any] | None = st.session_state.get("last_query_meta")
    if not meta:
        st.info("Ask a question in **Chat** to populate live retrieval stats here.")
    else:
        retrieval = meta.get("retrieval") or {}
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Corpus chunks", retrieval.get("corpus_size", "—"))
        c2.metric("Retrieved", retrieval.get("retrieved", "—"))
        c3.metric("Reranked", retrieval.get("reranked", "—"))
        c4.metric("Confidence", f"{float(meta.get('confidence') or 0):.2f}")

        st.write(
            f"Provider `{meta.get('provider', '?')}` · model `{meta.get('model', '?')}` · "
            f"refused **{meta.get('refused')}**"
        )

        citations = meta.get("citations") or []
        if citations:
            st.markdown("#### Top retrieved chunks")
            rows = [
                {
                    "rank": i,
                    "arxiv_id": c.get("arxiv_id"),
                    "pages": f"{c.get('page_start')}–{c.get('page_end')}",
                    "score": c.get("score"),
                    "title": c.get("title"),
                }
                for i, c in enumerate(citations, start=1)
            ]
            st.dataframe(rows, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("#### Harness report")
    if _REPORT_MD.exists():
        st.caption(f"Loaded `{_REPORT_MD}`")
        st.markdown(_REPORT_MD.read_text(encoding="utf-8"))
        if _REPORT_JSON.exists():
            with st.expander("Raw JSON"):
                st.code(_REPORT_JSON.read_text(encoding="utf-8"), language="json")
    else:
        st.info(
            "No report yet. From the repo root run:\n\n"
            "`python -m eval --report eval/reports/latest.md`\n\n"
            "Requires the demo paper ingested (`1706.03762`)."
        )
