"""Streamlit Community Cloud / Spaces entrypoint.

Starts the FastAPI backend in a background process so Chat/Admin/Eval can call
``http://127.0.0.1:8000``. Streamlit secrets are copied into ``os.environ``
before uvicorn starts so the API sees ``HF_TOKEN`` / provider keys.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import streamlit as st

# Repo root on sys.path (Streamlit may set cwd / script dir differently on Cloud).
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# UI → API on the same container.
os.environ.setdefault("API_BASE_URL", "http://127.0.0.1:8000")

_API_HOST = "127.0.0.1"
_API_PORT = 8000
_TMP = Path(tempfile.gettempdir())
_LOCK = _TMP / "arxiv_rag_api.lock"


def _apply_streamlit_secrets() -> None:
    """Copy root-level ``st.secrets`` into the process env for the API child."""

    try:
        secrets = st.secrets
    except Exception:  # noqa: BLE001 - no secrets.toml / Cloud secrets yet
        return
    for key in secrets:
        try:
            value = secrets[key]
        except Exception:  # noqa: BLE001
            continue
        if isinstance(value, (str, int, float, bool)):
            os.environ.setdefault(str(key), str(value))


def _port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def _ensure_api() -> None:
    """Start uvicorn once; Streamlit reruns must not spawn more workers."""

    if _port_open(_API_HOST, _API_PORT):
        return
    if _LOCK.exists():
        for _ in range(60):
            if _port_open(_API_HOST, _API_PORT):
                return
            time.sleep(0.5)
        return

    _LOCK.write_text(str(os.getpid()), encoding="utf-8")
    log_path = _TMP / "arxiv_rag_api.log"
    log_file = log_path.open("w", encoding="utf-8")
    # Inherit env so HF_TOKEN / LLM_* from Streamlit secrets reach the API.
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
        env=os.environ.copy(),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    for _ in range(120):
        if _port_open(_API_HOST, _API_PORT):
            return
        time.sleep(0.5)


_apply_streamlit_secrets()
_ensure_api()

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
        hf = health.get("hf_token")
        if hf:
            st.caption(f"HF token: {hf}")
        if health.get("langfuse"):
            st.caption(f"Langfuse: {health.get('langfuse')}")
    except ApiError as exc:
        st.error(str(exc))
    except Exception as exc:  # noqa: BLE001 - API may still be booting
        st.warning(f"API starting or unreachable: {exc}")
        st.caption("Wait a few seconds and refresh — the API boots beside Streamlit.")

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
