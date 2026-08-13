"""Tests for the Streamlit API client and citation markdown helpers."""

from __future__ import annotations

import json

import httpx
import pytest

from ui.client import ApiClient, ApiError, decode_sse_payload, parse_sse_data_line
from ui.render import citation_markdown, citations_markdown


def test_parse_sse_data_line() -> None:
    assert parse_sse_data_line('data: {"type": "token"}') == '{"type": "token"}'
    assert parse_sse_data_line("data: [DONE]") == "[DONE]"
    assert parse_sse_data_line(": keep-alive") is None
    assert parse_sse_data_line("") is None


def test_decode_sse_payload() -> None:
    assert decode_sse_payload("[DONE]") is None
    event = decode_sse_payload('{"type": "token", "text": "Hi"}')
    assert event == {"type": "token", "text": "Hi"}


def test_citation_markdown_includes_pages_and_link() -> None:
    md = citation_markdown(
        {
            "title": "Attention Is All You Need",
            "arxiv_id": "1706.03762v7",
            "page_start": 4,
            "page_end": 4,
            "score": 4.08,
            "source_url": "https://arxiv.org/pdf/1706.03762v7",
            "excerpt": "Multi-Head Attention consists of several attention layers",
        },
        index=1,
    )
    assert "[1]" in md
    assert "p.4–4" in md
    assert "arxiv.org" in md
    assert "Multi-Head" in md


def test_citations_markdown_empty() -> None:
    assert "No citations" in citations_markdown([])


def test_health_and_ingest_with_mock_transport() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok", "version": "0.1.0"})
        if request.url.path == "/ingest":
            body = json.loads(request.content)
            assert body["ids"] == ["1706.03762"]
            return httpx.Response(
                200,
                json={
                    "papers_fetched": 1,
                    "papers_indexed": 1,
                    "pages_parsed": 15,
                    "chunks_indexed": 22,
                    "collection_count": 22,
                    "skipped": [],
                },
            )
        return httpx.Response(404)

    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    api = ApiClient(base_url="http://test", client=http)
    assert api.health()["status"] == "ok"
    report = api.ingest(ids=["1706.03762"], max_results=1)
    assert report["chunks_indexed"] == 22


def test_stream_query_parses_meta_tokens_done() -> None:
    frames = (
        'data: {"type": "meta", "refused": false, "confidence": 1.5, "citations": []}\n\n'
        'data: {"type": "token", "text": "Hello"}\n\n'
        "data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/query"
        payload = json.loads(request.content)
        assert payload["stream"] is True
        assert payload["question"] == "hi"
        return httpx.Response(200, text=frames)

    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    api = ApiClient(base_url="http://test", client=http)
    events = list(api.stream_query("hi"))
    assert events[0]["type"] == "meta"
    assert events[1] == {"type": "token", "text": "Hello"}
    assert len(events) == 2


def test_api_error_on_http_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001
        return httpx.Response(502, json={"detail": "ingest failed"})

    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    api = ApiClient(base_url="http://test", client=http)
    with pytest.raises(ApiError) as err:
        api.health()
    assert err.value.status_code == 502
    assert "ingest failed" in err.value.detail
