"""Free Hugging Face Spaces entry (Gradio SDK + ZeroGPU).

Free Gradio Spaces run on ZeroGPU, which requires Gradio-bound handlers to use
``@spaces.GPU``. Do not name this file ``app.py`` (conflicts with ``app/`` package);
``app.py`` is a thin loader that imports this module.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import spaces

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

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
    if _port_open(_API_HOST, _API_PORT):
        return
    if _LOCK.exists():
        for _ in range(60):
            if _port_open(_API_HOST, _API_PORT):
                return
            time.sleep(0.5)
        return

    _LOCK.write_text(str(os.getpid()), encoding="utf-8")
    log_file = (_TMP / "arxiv_rag_api.log").open("w", encoding="utf-8")
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


def _health_markdown() -> str:
    from ui.client import ApiClient, ApiError, api_base_url

    _ensure_api()
    try:
        health = ApiClient().health()
        return (
            f"**API** `{api_base_url()}` — "
            f"`{health.get('status', '?')}` · v{health.get('version', '?')} · "
            f"langfuse `{health.get('langfuse', 'n/a')}`"
        )
    except ApiError as exc:
        return f"**API error:** {exc}"
    except Exception as exc:  # noqa: BLE001
        return f"**API starting…** ({exc})"


def _format_sources(citations: list[dict[str, Any]]) -> str:
    from ui.render import citations_markdown

    if not citations:
        return ""
    return "\n\n---\n### Sources\n\n" + citations_markdown(citations)


def chat(message: str, history: list[dict[str, str]]):
    """Stream an answer for ``gr.ChatInterface``.

    ChatInterface expects each yield to be the **assistant message string**
    (not a full history list).

    Intentionally **not** wrapped in ``@spaces.GPU``: chat only HTTP-calls the
    local FastAPI sidecar. Running it on a ZeroGPU worker burns quota and the
    worker often dies mid-SSE (``incomplete chunked read``). Ingest stays
    GPU-decorated so ZeroGPU still detects a bound handler at startup.
    """

    from ui.client import ApiClient, ApiError

    _ = history  # ChatInterface manages history; keep signature for Gradio.
    _ensure_api()
    yield "_Retrieving…_"

    client = ApiClient()

    def _render(payload: dict[str, Any]) -> str:
        answer = str(payload.get("answer") or "")
        citations = list(payload.get("citations") or [])
        if payload.get("refused"):
            prefix = "_Low confidence / refused._\n\n"
        elif payload.get("confidence") is not None:
            prefix = f"_confidence `{float(payload['confidence']):.3f}`_\n\n"
        else:
            prefix = ""
        return prefix + (answer or "_Empty answer._") + _format_sources(citations)

    answer = ""
    citations: list[dict[str, Any]] = []
    prefix = ""
    try:
        for event in client.stream_query(message):
            kind = event.get("type")
            if kind == "meta":
                citations = list(event.get("citations") or [])
                refused = bool(event.get("refused"))
                conf = event.get("confidence")
                if refused:
                    prefix = "_Low confidence / refused._\n\n"
                elif conf is not None:
                    prefix = f"_confidence `{float(conf):.3f}`_\n\n"
                yield prefix + (answer or "_Generating…_")
            elif kind == "token":
                answer += str(event.get("text") or "")
                yield prefix + answer + _format_sources(citations)
        if not answer:
            yield prefix + "_Empty answer._" + _format_sources(citations)
    except ApiError as exc:
        yield f"**API error:** {exc}"
    except Exception as stream_exc:  # noqa: BLE001 - SSE often dies on Spaces proxies
        yield "_Stream interrupted — retrying without streaming…_"
        try:
            yield _render(client.query(message))
        except Exception as fallback_exc:  # noqa: BLE001
            yield (
                f"**Error:** {stream_exc}\n\n" f"_Non-stream fallback also failed:_ {fallback_exc}"
            )


@spaces.GPU(duration=120)
def ingest(ids_text: str, query: str, max_results: float | int) -> str:
    """Ingest papers; ``@spaces.GPU`` so ZeroGPU detects a bound GPU handler."""

    from ui.client import ApiClient, ApiError

    _ensure_api()
    ids = [part.strip() for part in ids_text.replace(",", "\n").split() if part.strip()]
    q = (query or "").strip() or None
    if not ids and not q:
        return "Provide paper IDs and/or a search query."
    try:
        report = ApiClient().ingest(
            query=q,
            ids=ids or None,
            max_results=int(max_results),
        )
    except ApiError as exc:
        return f"**Ingest failed:** {exc}"
    except Exception as exc:  # noqa: BLE001
        return f"**Error:** {exc}"

    skipped = report.get("skipped") or []
    skip_md = ("\n\nSkipped:\n" + "\n".join(f"- {s}" for s in skipped)) if skipped else ""
    return (
        f"**Done.** fetched `{report.get('papers_fetched', 0)}` · "
        f"indexed `{report.get('papers_indexed', 0)}` · "
        f"chunks `{report.get('chunks_indexed', 0)}` · "
        f"collection `{report.get('collection_count', 0)}`"
        f"{skip_md}"
    )


def build_demo():
    import gradio as gr

    with gr.Blocks(title="arXiv RAG Assistant") as demo:
        gr.Markdown(
            "# arXiv RAG Assistant\n"
            "Ask questions about ingested arXiv AI/ML papers (answers cite sources)."
        )
        status = gr.Markdown(_health_markdown())
        refresh = gr.Button("Refresh API status", size="sm")
        refresh.click(_health_markdown, outputs=status)

        with gr.Tab("Chat"):
            gr.ChatInterface(fn=chat, type="messages")

        with gr.Tab("Admin — ingest"):
            gr.Markdown("Fetch papers from arXiv, chunk, embed, and index in Chroma.")
            ids_box = gr.Textbox(
                label="arXiv IDs (one per line or comma-separated)",
                placeholder="1706.03762",
                lines=3,
            )
            query_box = gr.Textbox(
                label="Or search query",
                placeholder="retrieval augmented generation",
            )
            max_box = gr.Number(label="Max papers", value=1, minimum=1, maximum=20, precision=0)
            go = gr.Button("Ingest", variant="primary")
            result = gr.Markdown()
            go.click(ingest, inputs=[ids_box, query_box, max_box], outputs=result)

        gr.Markdown(
            "_Free Gradio ZeroGPU Space: FastAPI runs beside this UI. "
            "Set `HUGGINGFACE_API_KEY` in Space secrets for the default LLM._"
        )
    return demo


_ensure_api()
demo = build_demo()

if __name__ == "__main__":
    demo.launch(ssr_mode=False)
