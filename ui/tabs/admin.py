"""Admin tab: ingest papers into the vector store."""

from __future__ import annotations

import streamlit as st

from ui.client import ApiClient, ApiError


def render_admin(client: ApiClient) -> None:
    """Form to ingest by arXiv search query or explicit paper IDs."""

    st.subheader("Admin — ingest")
    st.caption("Fetch papers from arXiv, chunk them, embed, and index in Chroma.")

    mode = st.radio("Source", ["Paper IDs", "Search query"], horizontal=True)
    max_results = st.number_input("Max papers", min_value=1, max_value=50, value=1, step=1)

    query: str | None = None
    ids: list[str] | None = None
    if mode == "Paper IDs":
        raw = st.text_area(
            "arXiv IDs (one per line or comma-separated)",
            placeholder="1706.03762\n2005.11401",
        )
        ids = [part.strip() for part in raw.replace(",", "\n").split() if part.strip()]
    else:
        query = st.text_input("arXiv search query", placeholder="retrieval augmented generation")

    if st.button("Ingest", type="primary"):
        if mode == "Paper IDs" and not ids:
            st.error("Enter at least one arXiv ID.")
            return
        if mode == "Search query" and not (query or "").strip():
            st.error("Enter a search query.")
            return
        with st.spinner("Ingesting… this can take a few minutes (PDF download + embeddings)."):
            try:
                report = client.ingest(query=query or None, ids=ids, max_results=int(max_results))
            except ApiError as exc:
                st.error(str(exc))
                return
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))
                return

        st.success("Ingestion complete.")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Fetched", report.get("papers_fetched", 0))
        c2.metric("Indexed", report.get("papers_indexed", 0))
        c3.metric("Chunks", report.get("chunks_indexed", 0))
        c4.metric("Collection size", report.get("collection_count", 0))
        skipped = report.get("skipped") or []
        if skipped:
            st.warning("Skipped:\n" + "\n".join(f"- {item}" for item in skipped))
        st.session_state.last_ingest_report = report
