"""Streamlit UI: Chat, Admin (ingest), and Eval tabs over the FastAPI backend."""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit puts the script directory (`ui/`) on sys.path, not the repo root.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402

from ui.client import ApiClient, ApiError, api_base_url  # noqa: E402
from ui.tabs.admin import render_admin  # noqa: E402
from ui.tabs.chat import render_chat  # noqa: E402
from ui.tabs.eval import render_eval  # noqa: E402

st.set_page_config(page_title="arXiv RAG Assistant", page_icon="📄", layout="wide")
st.title("📄 arXiv RAG Assistant")
st.caption("Ask questions about arXiv AI/ML papers with inline citations.")

client = ApiClient()

with st.sidebar:
    st.header("Backend")
    st.code(api_base_url(), language="text")
    try:
        health = client.health()
        st.success(f"API {health.get('status', 'unknown')} · v{health.get('version', '?')}")
        st.caption(health.get("environment", ""))
    except ApiError as exc:
        st.error(str(exc))
    except Exception as exc:  # noqa: BLE001 - connection errors are expected when API is down
        st.error(f"API unreachable: {exc}")
        st.caption("Start the API: `uvicorn app.api.main:app --reload`")

    if st.button("Clear chat"):
        st.session_state.chat_history = []
        st.session_state.last_query_meta = None
        st.rerun()

chat_tab, admin_tab, eval_tab = st.tabs(["Chat", "Admin", "Eval"])
with chat_tab:
    render_chat(client)
with admin_tab:
    render_admin(client)
with eval_tab:
    render_eval(client)
