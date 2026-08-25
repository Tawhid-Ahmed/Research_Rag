"""Tests for token counting, cost estimates, and observability wiring."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

import app.api.deps as deps
from app.api.main import app
from app.config import Settings, get_settings
from app.llm.types import ChatMessage
from app.observability.cost import estimate_cost_usd
from app.observability.tokens import count_message_tokens, count_text_tokens
from app.observability.tracing import (
    NoOpObservability,
    UsageStats,
    build_observability,
    compute_usage,
)
from tests.test_api_endpoints import FakeLLM, FakeRetriever, _strong_hit


def test_count_text_tokens_non_empty() -> None:
    assert count_text_tokens("hello world") > 0
    assert count_text_tokens("") == 0


def test_count_message_tokens_includes_roles() -> None:
    messages = [
        ChatMessage(role="system", content="Be helpful."),
        ChatMessage(role="user", content="What is attention?"),
    ]
    assert count_message_tokens(messages) > count_text_tokens("What is attention?")


def test_estimate_cost_openai_positive_ollama_zero() -> None:
    openai_cost = estimate_cost_usd(
        provider="openai",
        model="gpt-4o-mini",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )
    assert openai_cost == 0.15 + 0.60
    ollama_cost = estimate_cost_usd(
        provider="ollama",
        model="gemma4:e4b",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )
    assert ollama_cost == 0.0


def test_compute_usage_totals() -> None:
    usage = compute_usage(
        messages=[ChatMessage(role="user", content="hi")],
        answer="hello there",
        provider="ollama",
        model="gemma",
    )
    assert isinstance(usage, UsageStats)
    assert usage.total_tokens == usage.input_tokens + usage.output_tokens
    assert usage.estimated_cost_usd == 0.0
    assert "input_tokens" in usage.as_dict()


def test_build_observability_noop_without_keys() -> None:
    obs = build_observability(Settings(langfuse_public_key=None, langfuse_secret_key=None))
    assert isinstance(obs, NoOpObservability)
    assert obs.enabled is False
    trace = obs.start_query("q", provider="ollama", model="x")
    trace.record_retrieval(
        retrieval={"corpus_size": 1, "retrieved": 1, "reranked": 1},
        confidence=1.0,
        refused=False,
        citation_count=1,
    )
    trace.record_generation(
        messages=[],
        answer="a",
        provider="ollama",
        model="x",
        usage=UsageStats(),
    )
    trace.end(output="a")


def test_build_observability_enabled_with_keys(monkeypatch) -> None:
    created: dict[str, Any] = {}

    class FakeLangfuse:
        def __init__(self, **kwargs: Any) -> None:
            created.update(kwargs)

        def trace(self, **kwargs: Any) -> Any:
            class T:
                def span(self, **kw: Any) -> None:
                    return None

                def generation(self, **kw: Any) -> None:
                    return None

                def update(self, **kw: Any) -> None:
                    return None

            return T()

        def flush(self) -> None:
            return None

    monkeypatch.setattr("langfuse.Langfuse", FakeLangfuse)
    obs = build_observability(
        Settings(
            langfuse_public_key="pk-test",
            langfuse_secret_key="sk-test",
            langfuse_host="https://example.com",
        )
    )
    assert obs.enabled is True
    assert created["public_key"] == "pk-test"
    trace = obs.start_query("q", provider="openai", model="gpt-4o-mini")
    usage = UsageStats(input_tokens=10, output_tokens=5, estimated_cost_usd=0.001)
    trace.record_retrieval(
        retrieval={"corpus_size": 1, "retrieved": 1, "reranked": 1},
        confidence=1.0,
        refused=False,
        citation_count=1,
    )
    trace.record_generation(
        messages=[ChatMessage(role="user", content="q")],
        answer="a",
        provider="openai",
        model="gpt-4o-mini",
        usage=usage,
    )
    trace.end(output="a")


def test_health_reports_langfuse_disabled() -> None:
    client = TestClient(app)
    body = client.get("/health").json()
    assert body["langfuse"] == "disabled"


def test_query_includes_usage(monkeypatch) -> None:
    monkeypatch.setattr(deps, "_retriever_factory", lambda settings: FakeRetriever([_strong_hit()]))
    monkeypatch.setattr(deps, "_llm_factory", lambda settings: FakeLLM())
    app.dependency_overrides[get_settings] = lambda: Settings(
        min_confidence_score=0.2,
        llm_provider="ollama",
        llm_model="gemma",
    )
    try:
        client = TestClient(app)
        body = client.post("/query", json={"question": "What is attention?"}).json()
        assert body["usage"] is not None
        assert body["usage"]["total_tokens"] > 0
        assert body["usage"]["estimated_cost_usd"] == 0.0
    finally:
        app.dependency_overrides.clear()


def test_traced_token_stream_emits_usage_before_done() -> None:
    import asyncio
    import json

    from app.api.main import _traced_token_stream
    from app.llm.types import ChatMessage, GenerationConfig
    from app.observability.tracing import NoOpObservability

    llm = FakeLLM("Hello world")
    messages = [ChatMessage(role="user", content="hi")]
    trace = NoOpObservability().start_query("hi", provider="ollama", model="gemma")

    async def _run() -> list[str]:
        frames = [
            frame
            async for frame in _traced_token_stream(
                llm=llm,
                messages=messages,
                gen_config=GenerationConfig(),
                meta={"confidence": 1.0},
                trace=trace,
                provider="ollama",
                model="gemma",
            )
        ]
        return [f["data"] for f in frames]

    payloads = asyncio.run(_run())
    assert payloads[-1] == "[DONE]"
    usage_payload = json.loads(payloads[-2])
    assert usage_payload["type"] == "usage"
    assert usage_payload["total_tokens"] > 0
