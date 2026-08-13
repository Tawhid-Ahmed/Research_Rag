"""Hugging Face Inference API provider.

Uses ``huggingface_hub.InferenceClient`` with lazy import. Streaming prefers
token-level ``text_generation(..., stream=True)`` and falls back to a single
full-text chunk if the endpoint does not support streaming.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator

from app.llm.providers.base import messages_to_prompt
from app.llm.types import ChatMessage, GenerationConfig, StreamChunk


class HuggingFaceProvider:
    """Chat provider backed by Hugging Face Inference API."""

    def __init__(
        self,
        *,
        model_name: str,
        api_key: str | None,
        temperature: float,
        max_tokens: int,
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client: object | None = None

    def _get_client(self) -> object:
        if self._client is None:
            from huggingface_hub import InferenceClient

            self._client = InferenceClient(model=self.model_name, token=self.api_key)
        return self._client

    def _generate_sync(self, messages: list[ChatMessage], config: GenerationConfig) -> str:
        client = self._get_client()
        prompt = messages_to_prompt(messages)
        result = client.text_generation(  # type: ignore[attr-defined]
            prompt,
            max_new_tokens=config.max_tokens,
            temperature=config.temperature,
            stream=False,
        )
        return str(result)

    def _stream_sync(self, messages: list[ChatMessage], config: GenerationConfig) -> Iterator[str]:
        client = self._get_client()
        prompt = messages_to_prompt(messages)
        try:
            stream = client.text_generation(  # type: ignore[attr-defined]
                prompt,
                max_new_tokens=config.max_tokens,
                temperature=config.temperature,
                stream=True,
            )
            for token in stream:
                text = getattr(getattr(token, "token", None), "text", None)
                if text is None:
                    text = str(token)
                if text:
                    yield text
        except Exception:
            # Some hosted models reject stream=True; degrade to one chunk.
            yield self._generate_sync(messages, config)

    async def generate(self, messages: list[ChatMessage], config: GenerationConfig) -> str:
        return await asyncio.to_thread(self._generate_sync, messages, config)

    async def stream_generate(
        self, messages: list[ChatMessage], config: GenerationConfig
    ) -> AsyncIterator[StreamChunk]:
        """Yield token chunks, then a final ``done=True`` marker.

        Tokens are collected in a worker thread (HF client is sync) then
        yielded to the async caller so the SSE layer can flush promptly.
        """

        pieces = await asyncio.to_thread(lambda: list(self._stream_sync(messages, config)))
        for piece in pieces:
            yield StreamChunk(text=piece)
        yield StreamChunk(text="", done=True)
