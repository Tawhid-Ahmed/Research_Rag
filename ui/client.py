"""HTTP client for the FastAPI backend.

Kept free of Streamlit imports so it is unit-testable with ``httpx.MockTransport``.
The Chat tab streams SSE; Admin uses JSON ingest; the sidebar uses ``/health``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Any

import httpx

DEFAULT_API_BASE_URL = "http://localhost:8000"
DONE_SENTINEL = "[DONE]"


def api_base_url() -> str:
    """Return the API origin from ``API_BASE_URL`` (compose sets this for the UI)."""

    return os.getenv("API_BASE_URL", DEFAULT_API_BASE_URL).rstrip("/")


def parse_sse_data_line(line: str) -> str | None:
    """Extract the payload from a ``data: ...`` SSE line, or ``None`` if not data."""

    stripped = line.strip()
    if not stripped.startswith("data:"):
        return None
    return stripped[5:].lstrip()


def decode_sse_payload(payload: str) -> dict[str, Any] | None:
    """Decode one SSE data payload.

    Returns ``None`` for the ``[DONE]`` sentinel; otherwise a JSON object
    (``meta`` / ``token`` events from ``POST /query?stream=true``).
    """

    if payload == DONE_SENTINEL:
        return None
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in SSE payload, got {type(data).__name__}")
    return data


class ApiError(RuntimeError):
    """Raised when the backend returns a non-success HTTP status."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"API {status_code}: {detail}")


class ApiClient:
    """Thin wrapper around health / ingest / query."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        client: httpx.Client | None = None,
        timeout: float = 120.0,
        ingest_timeout: float = 600.0,
    ) -> None:
        self.base_url = (base_url or api_base_url()).rstrip("/")
        self.ingest_timeout = ingest_timeout
        self._owns_client = client is None
        self._http = client or httpx.Client(base_url=self.base_url, timeout=timeout)

    def close(self) -> None:
        if self._owns_client:
            self._http.close()

    def health(self) -> dict[str, Any]:
        response = self._http.get("/health")
        self._raise_for_status(response)
        return response.json()

    def ingest(
        self,
        *,
        query: str | None = None,
        ids: list[str] | None = None,
        max_results: int = 10,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"max_results": max_results}
        if query:
            payload["query"] = query
        if ids:
            payload["ids"] = ids
        response = self._http.post("/ingest", json=payload, timeout=self.ingest_timeout)
        self._raise_for_status(response)
        return response.json()

    def query(self, question: str, *, top_k: int | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"question": question, "stream": False}
        if top_k is not None:
            payload["top_k"] = top_k
        response = self._http.post("/query", json=payload)
        self._raise_for_status(response)
        return response.json()

    def stream_query(self, question: str, *, top_k: int | None = None) -> Iterator[dict[str, Any]]:
        """Yield ``meta`` / ``token`` event dicts from the SSE query endpoint."""

        payload: dict[str, Any] = {"question": question, "stream": True}
        if top_k is not None:
            payload["top_k"] = top_k
        with self._http.stream(
            "POST", "/query", json=payload, timeout=self.ingest_timeout
        ) as response:
            self._raise_for_status(response)
            for line in response.iter_lines():
                data = parse_sse_data_line(line)
                if data is None:
                    continue
                event = decode_sse_payload(data)
                if event is None:
                    return
                yield event

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        if response.is_success:
            return
        detail = response.text
        try:
            body = response.json()
            if isinstance(body, dict) and "detail" in body:
                detail = str(body["detail"])
        except Exception:  # noqa: BLE001 - fall back to raw text
            pass
        raise ApiError(response.status_code, detail)
