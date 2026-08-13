"""Unit tests for provider streaming and SSE encoding."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator

from app.llm.providers.huggingface import HuggingFaceProvider
from app.llm.providers.openai import OpenAIProvider
from app.llm.sse import DONE_SENTINEL, encode_sse_chunk, iter_sse
from app.llm.types import ChatMessage, GenerationConfig, StreamChunk


def test_encode_sse_chunk_text_and_done() -> None:
    text_line = encode_sse_chunk(StreamChunk(text="hello"))
    assert text_line == 'data: {"text": "hello"}\n\n'
    done_line = encode_sse_chunk(StreamChunk(done=True))
    assert done_line == f"data: {DONE_SENTINEL}\n\n"


def test_iter_sse_appends_done_if_missing() -> None:
    async def source() -> AsyncIterator[StreamChunk]:
        yield StreamChunk(text="a")
        yield StreamChunk(text="b")

    async def _run() -> list[str]:
        return [line async for line in iter_sse(source())]

    lines = asyncio.run(_run())
    assert lines[0] == 'data: {"text": "a"}\n\n'
    assert lines[1] == 'data: {"text": "b"}\n\n'
    assert lines[-1] == f"data: {DONE_SENTINEL}\n\n"


def test_huggingface_stream_generate_yields_tokens_then_done(monkeypatch) -> None:
    provider = HuggingFaceProvider(
        model_name="dummy-model",
        api_key=None,
        temperature=0.1,
        max_tokens=64,
    )

    def fake_stream(
        messages: list[ChatMessage],
        config: GenerationConfig,
    ) -> Iterator[str]:
        yield "hello"
        yield "-"
        yield "stream"

    monkeypatch.setattr(provider, "_stream_sync", fake_stream)

    messages = [ChatMessage(role="user", content="test")]
    config = GenerationConfig(temperature=0.1, max_tokens=10)

    async def _run() -> list[StreamChunk]:
        return [chunk async for chunk in provider.stream_generate(messages, config)]

    chunks = asyncio.run(_run())
    assert [c.text for c in chunks if not c.done] == ["hello", "-", "stream"]
    assert chunks[-1].done is True


def test_openai_stream_generate_yields_deltas(monkeypatch) -> None:
    provider = OpenAIProvider(
        model_name="gpt-test",
        api_key="sk-test",
        temperature=0.1,
        max_tokens=32,
    )

    class _Delta:
        def __init__(self, content: str | None) -> None:
            self.content = content

    class _Choice:
        def __init__(self, content: str | None) -> None:
            self.delta = _Delta(content)

    class _Event:
        def __init__(self, content: str | None) -> None:
            self.choices = [_Choice(content)]

    class _Stream:
        def __aiter__(self) -> _Stream:
            self._items = iter([_Event("Hi"), _Event("!"), _Event(None)])
            return self

        async def __anext__(self) -> _Event:
            try:
                return next(self._items)
            except StopIteration as exc:
                raise StopAsyncIteration from exc

    class _Completions:
        async def create(self, **kwargs: object) -> _Stream:
            assert kwargs.get("stream") is True
            return _Stream()

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    monkeypatch.setattr(provider, "_get_client", lambda: _Client())

    messages = [ChatMessage(role="user", content="hi")]
    config = GenerationConfig()

    async def _run() -> list[StreamChunk]:
        return [chunk async for chunk in provider.stream_generate(messages, config)]

    chunks = asyncio.run(_run())
    assert [c.text for c in chunks if not c.done] == ["Hi", "!"]
    assert chunks[-1].done is True
