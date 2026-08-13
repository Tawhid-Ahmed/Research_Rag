"""Chat tab: streaming Q&A with citations."""

from __future__ import annotations

from typing import Any

import streamlit as st

from ui.client import ApiClient, ApiError
from ui.render import citations_markdown


def render_chat(client: ApiClient) -> None:
    """Render chat history and a streaming question box."""

    st.subheader("Chat")
    st.caption(
        "Answers are grounded in ingested papers. Sources appear as soon as retrieval finishes."
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "last_query_meta" not in st.session_state:
        st.session_state.last_query_meta = None

    for turn in st.session_state.chat_history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn.get("citations"):
                with st.expander("Sources", expanded=turn.get("role") == "assistant"):
                    st.markdown(citations_markdown(turn["citations"]))
            if turn.get("refused"):
                st.warning("Low retrieval confidence — the API refused to guess.")

    prompt = st.chat_input("Ask about an ingested paper…")
    if not prompt:
        return

    st.session_state.chat_history.append({"role": "user", "content": prompt, "citations": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        status = st.status("Retrieving passages…", expanded=False)
        answer = ""
        citations: list[dict[str, Any]] = []
        refused = False
        meta: dict[str, Any] = {}
        try:
            for event in client.stream_query(prompt):
                kind = event.get("type")
                if kind == "meta":
                    citations = list(event.get("citations") or [])
                    refused = bool(event.get("refused"))
                    meta = event
                    st.session_state.last_query_meta = event
                    retrieval = event.get("retrieval") or {}
                    status.update(
                        label=(
                            f"Retrieved {retrieval.get('retrieved', '?')} · "
                            f"reranked {retrieval.get('reranked', '?')} · "
                            f"confidence {event.get('confidence', 0):.2f}"
                        ),
                        state="running",
                    )
                    if citations:
                        with st.expander("Sources", expanded=True):
                            st.markdown(citations_markdown(citations))
                elif kind == "token":
                    answer += str(event.get("text") or "")
                    placeholder.markdown(answer + " ▍")
            placeholder.markdown(answer or "_Empty answer._")
            status.update(label="Done", state="complete")
        except ApiError as exc:
            status.update(label="Request failed", state="error")
            placeholder.error(str(exc))
            answer = f"**Error:** {exc}"
        except Exception as exc:  # noqa: BLE001 - show any client/network failure
            status.update(label="Request failed", state="error")
            placeholder.error(str(exc))
            answer = f"**Error:** {exc}"

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": answer,
            "citations": citations,
            "refused": refused,
            "meta": meta,
        }
    )
