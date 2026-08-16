"""Hugging Face Inference Providers chat backend.

Uses the OpenAI-compatible chat-completions path via ``InferenceClient``.
The legacy ``api-inference.huggingface.co`` host and many bare
``text_generation`` models are retired; Spaces need a token with
**Make calls to Inference Providers** permission (``HF_TOKEN`` /
``HUGGINGFACE_API_KEY``).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from typing import Any

from app.llm.types import ChatMessage, GenerationConfig, StreamChunk


class HuggingFaceProvider:
    """Chat provider backed by Hugging Face Inference Providers."""

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

    def _require_api_key(self) -> str:
        if not self.api_key:
            raise ValueError(
                "HUGGINGFACE_API_KEY or HF_TOKEN is required for Hugging Face "
                "Inference Providers. Create a fine-grained token with "
                "'Make calls to Inference Providers' and set it as a Space secret."
            )
        return self.api_key

    def _get_client(self) -> Any:
        if self._client is None:
            from huggingface_hub import InferenceClient

            self._client = InferenceClient(token=self._require_api_key())
        return self._client

    @staticmethod
    def _to_dicts(messages: list[ChatMessage]) -> list[dict[str, str]]:
        return [{"role": msg.role, "content": msg.content} for msg in messages]

    def _generate_sync(self, messages: list[ChatMessage], config: GenerationConfig) -> str:
        client = self._get_client()
        result = client.chat_completion(
            self._to_dicts(messages),
            model=self.model_name,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            stream=False,
        )
        content = result.choices[0].message.content
        return str(content or "")

    def _stream_sync(self, messages: list[ChatMessage], config: GenerationConfig) -> Iterator[str]:
        client = self._get_client()
        try:
            stream = client.chat_completion(
                self._to_dicts(messages),
                model=self.model_name,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                stream=True,
            )
            for event in stream:
                choices = getattr(event, "choices", None) or []
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                text = getattr(delta, "content", None) if delta is not None else None
                if text:
                    yield str(text)
        except Exception:
            # Some providers reject stream=True; degrade to one chunk.
            text = self._generate_sync(messages, config)
            if text:
                yield text

    async def generate(self, messages: list[ChatMessage], config: GenerationConfig) -> str:
        return await asyncio.to_thread(self._generate_sync, messages, config)

    async def stream_generate(
        self, messages: list[ChatMessage], config: GenerationConfig
    ) -> AsyncIterator[StreamChunk]:
        """Yield token chunks, then a final ``done=True`` marker."""

        pieces = await asyncio.to_thread(lambda: list(self._stream_sync(messages, config)))
        for piece in pieces:
            yield StreamChunk(text=piece)
        yield StreamChunk(text="", done=True)
