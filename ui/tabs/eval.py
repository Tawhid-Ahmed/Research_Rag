"""Eval tab: live retrieval stats plus a placeholder for the Step 7 harness."""

from __future__ import annotations

from typing import Any

import streamlit as st

from ui.client import ApiClient


def render_eval(_client: ApiClient) -> None:
    """Show last-query retrieval numbers; RAGAS report lands in Step 7."""

    st.subheader("Eval")
    st.caption(
        "The full eval harness (golden Q/A, recall@k, MRR, RAGAS) is Step 7. "
        "Until then, this tab surfaces stats from the last Chat query."
    )

    meta: dict[str, Any] | None = st.session_state.get("last_query_meta")
    if not meta:
        st.info("Ask a question in **Chat** to populate retrieval stats here.")
        return

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
    st.markdown(
        """
**Coming in Step 7**

- Golden question set over ingested papers
- Retrieval metrics: recall@k, MRR, hit-rate
- RAGAS answer quality
- Before/after ablation (dense-only vs hybrid + rerank)
        """
    )
