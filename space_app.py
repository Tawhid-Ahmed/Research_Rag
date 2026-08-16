"""Free Hugging Face Spaces entrypoint (Streamlit SDK).

Starts the FastAPI backend in-process (background) so the existing Chat/Admin/Eval
UI can call ``http://127.0.0.1:8000``. Use this when Docker Spaces are unavailable
on the free tier.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Repo root on sys.path (Streamlit may set cwd / script dir differently on Spaces).
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# UI → API on the same Space container.
os.environ.setdefault("API_BASE_URL", "http://127.0.0.1:8000")

_API_HOST = "127.0.0.1"
_API_PORT = 8000
_TMP = Path(tempfile.gettempdir())
_LOCK = _TMP / "arxiv_rag_api.lock"


def _port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def _ensure_api() -> None:
    """Start uvicorn once; Streamlit reruns must not spawn more workers."""

    if _port_open(_API_HOST, _API_PORT):
        return
    if _LOCK.exists():
        # Another rerun already started it; wait briefly.
        for _ in range(60):
            if _port_open(_API_HOST, _API_PORT):
                return
            time.sleep(0.5)
        return

    _LOCK.write_text(str(os.getpid()), encoding="utf-8")
    log_path = _TMP / "arxiv_rag_api.log"
    log_file = log_path.open("w", encoding="utf-8")
    subprocess.Popen(  # noqa: S603 - fixed argv, no shell
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.api.main:app",
            "--host",
            _API_HOST,
            "--port",
            str(_API_PORT),
        ],
        cwd=str(_ROOT),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    for _ in range(120):
        if _port_open(_API_HOST, _API_PORT):
            return
        time.sleep(0.5)


_ensure_api()

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
        if health.get("langfuse"):
            st.caption(f"Langfuse: {health.get('langfuse')}")
    except ApiError as exc:
        st.error(str(exc))
    except Exception as exc:  # noqa: BLE001 - API may still be booting on Spaces
        st.warning(f"API starting or unreachable: {exc}")
        st.caption("Wait a few seconds and refresh — the API boots beside Streamlit on this Space.")

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
