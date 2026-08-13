"""Integration-style tests for /ingest and /query with fakes."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

from fastapi.testclient import TestClient

import app.api.deps as deps
from app.api.main import app
from app.config import Settings
from app.ingest.models import Chunk, IngestionReport
from app.llm.types import ChatMessage, GenerationConfig, StreamChunk
from app.retrieval.models import Hit, RetrievalResult


class FakeIngest:
    def run(
        self,
        *,
        query: str | None = None,
        ids: list[str] | None = None,
        max_results: int = 10,
    ) -> IngestionReport:
        return IngestionReport(
            papers_fetched=1,
            papers_indexed=1,
            pages_parsed=2,
            chunks_indexed=3,
            collection_count=3,
            skipped=[],
        )


class FakeRetriever:
    def __init__(self, hits: list[Hit]) -> None:
        self._hits = hits

    def search(self, query: str) -> RetrievalResult:
        return RetrievalResult(
            query=query,
            hits=self._hits,
            corpus_size=10,
            retrieved=len(self._hits),
            reranked=len(self._hits),
        )


class FakeLLM:
    def __init__(self, answer: str = "Attention is useful [1].") -> None:
        self.answer = answer
        self.calls: list[list[ChatMessage]] = []

    async def generate(self, messages: list[ChatMessage], config: GenerationConfig) -> str:
        self.calls.append(messages)
        return self.answer

    async def stream_generate(
        self, messages: list[ChatMessage], config: GenerationConfig
    ) -> AsyncIterator[StreamChunk]:
        self.calls.append(messages)
        for piece in ["Attention ", "is ", "useful."]:
            yield StreamChunk(text=piece)
        yield StreamChunk(done=True)


def _strong_hit() -> Hit:
    chunk = Chunk(
        chunk_id="1706.03762::0",
        arxiv_id="1706.03762",
        title="Attention Is All You Need",
        text="Multi-head attention lets the model attend to subspaces.",
        token_count=10,
        chunk_index=0,
        page_start=3,
        page_end=4,
        source_url="https://arxiv.org/pdf/1706.03762",
    )
    return Hit(chunk=chunk, score=0.9, rerank_score=0.9)


def _weak_hit() -> Hit:
    hit = _strong_hit()
    return Hit(chunk=hit.chunk, score=0.05, rerank_score=0.05)


def test_ingest_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(deps, "_ingest_factory", lambda settings: FakeIngest())
    client = TestClient(app)
    response = client.post("/ingest", json={"ids": ["1706.03762"], "max_results": 1})
    assert response.status_code == 200
    body = response.json()
    assert body["papers_indexed"] == 1
    assert body["chunks_indexed"] == 3


def test_ingest_requires_query_or_ids() -> None:
    client = TestClient(app)
    response = client.post("/ingest", json={"max_results": 1})
    assert response.status_code == 422


def test_query_returns_answer_and_citations(monkeypatch) -> None:
    monkeypatch.setattr(deps, "_retriever_factory", lambda settings: FakeRetriever([_strong_hit()]))
    monkeypatch.setattr(deps, "_llm_factory", lambda settings: FakeLLM())
    monkeypatch.setattr(
        deps,
        "get_settings",
        lambda: Settings(min_confidence_score=0.2, llm_provider="ollama", llm_model="gemma"),
    )
    # FastAPI Depends(get_settings) uses the real function from app.config — override app dep.
    from app.config import get_settings as real_get_settings

    app.dependency_overrides[real_get_settings] = lambda: Settings(
        min_confidence_score=0.2,
        llm_provider="ollama",
        llm_model="gemma",
    )
    try:
        client = TestClient(app)
        response = client.post("/query", json={"question": "What is attention?"})
        assert response.status_code == 200
        body = response.json()
        assert body["refused"] is False
        assert "Attention" in body["answer"]
        assert body["citations"][0]["arxiv_id"] == "1706.03762"
        assert body["citations"][0]["page_start"] == 3
        assert body["confidence"] >= 0.2
        assert body["provider"] == "ollama"
    finally:
        app.dependency_overrides.clear()


def test_query_refuses_low_confidence(monkeypatch) -> None:
    monkeypatch.setattr(deps, "_retriever_factory", lambda settings: FakeRetriever([_weak_hit()]))
    monkeypatch.setattr(deps, "_llm_factory", lambda settings: FakeLLM("should not be used"))
    from app.config import get_settings as real_get_settings

    app.dependency_overrides[real_get_settings] = lambda: Settings(min_confidence_score=0.2)
    try:
        client = TestClient(app)
        response = client.post("/query", json={"question": "Unrelated?"})
        assert response.status_code == 200
        body = response.json()
        assert body["refused"] is True
        assert "could not find sufficiently relevant" in body["answer"].lower()
    finally:
        app.dependency_overrides.clear()


def test_query_stream_emits_meta_tokens_and_done(monkeypatch) -> None:
    monkeypatch.setattr(deps, "_retriever_factory", lambda settings: FakeRetriever([_strong_hit()]))
    monkeypatch.setattr(deps, "_llm_factory", lambda settings: FakeLLM())
    from app.config import get_settings as real_get_settings

    app.dependency_overrides[real_get_settings] = lambda: Settings(
        min_confidence_score=0.2,
        llm_provider="ollama",
        llm_model="gemma",
    )
    try:
        client = TestClient(app)
        with client.stream(
            "POST", "/query", json={"question": "What is attention?", "stream": True}
        ) as response:
            assert response.status_code == 200
            raw = "".join(response.iter_text())
        assert '"type": "meta"' in raw or '"type":"meta"' in raw
        assert "token" in raw
        assert "[DONE]" in raw
        # Parse first data payload
        lines = [line[6:] for line in raw.splitlines() if line.startswith("data: ")]
        meta = json.loads(lines[0])
        assert meta["type"] == "meta"
        assert meta["refused"] is False
        assert meta["citations"][0]["chunk_id"] == "1706.03762::0"
    finally:
        app.dependency_overrides.clear()


def test_streaming_helper_encodes_events() -> None:
    from app.api.streaming import iter_query_sse
    from app.llm.types import StreamChunk

    async def tokens() -> AsyncIterator[StreamChunk]:
        yield StreamChunk(text="Hi")
        yield StreamChunk(done=True)

    async def _run() -> list[dict[str, str]]:
        return [x async for x in iter_query_sse(meta={"refused": False}, token_stream=tokens())]

    frames = asyncio.run(_run())
    assert json.loads(frames[0]["data"])["type"] == "meta"
    assert json.loads(frames[1]["data"])["text"] == "Hi"
    assert frames[-1]["data"] == "[DONE]"
