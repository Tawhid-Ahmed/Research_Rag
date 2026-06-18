"""Streamlit UI entrypoint (scaffold).

Full Chat / Admin / Eval tabs are implemented in a later build step. This stub
renders a placeholder and a backend health check so the container and demo boot.
"""

from __future__ import annotations

import os

import httpx
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="arXiv RAG Assistant", page_icon="📄", layout="wide")
st.title("📄 arXiv RAG Assistant")
st.caption("Ask questions about arXiv AI/ML papers with inline citations.")

with st.sidebar:
    st.header("Status")
    try:
        resp = httpx.get(f"{API_BASE_URL}/health", timeout=5.0)
        resp.raise_for_status()
        st.success(f"API: {resp.json().get('status', 'unknown')}")
    except Exception as exc:  # noqa: BLE001 - surface any connection error to the user
        st.error(f"API unreachable: {exc}")

st.info("UI scaffold in place. Chat, Admin, and Eval tabs are coming in later build steps.")
