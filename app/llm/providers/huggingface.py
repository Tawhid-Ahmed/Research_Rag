"""Hugging Face Inference provider (Inference Providers / router).

The legacy host ``api-inference.huggingface.co`` was retired. Requests now go
through ``https://router.huggingface.co/hf-inference/models/<model>``. We pass
that full URL into ``InferenceClient`` so older ``huggingface_hub`` versions
(still shipping the old default base) keep working.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator

from app.llm.providers.base import messages_to_prompt
from app.llm.types import ChatMessage, GenerationConfig, StreamChunk

HF_INFERENCE_ROUTER = "https://router.huggingface.co/hf-inference/models"


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

    @property
    def model_url(self) -> str:
        """Absolute router URL for this model (avoids the retired API host)."""

        if self.model_name.startswith("http://") or self.model_name.startswith("https://"):
            return self.model_name
        return f"{HF_INFERENCE_ROUTER}/{self.model_name}"

    def _get_client(self) -> object:
        if self._client is None:
            from huggingface_hub import InferenceClient

            self._client = InferenceClient(model=self.model_url, token=self.api_key)
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
        """Yield token chunks, then a final ``done=True`` marker."""

        pieces = await asyncio.to_thread(lambda: list(self._stream_sync(messages, config)))
        for piece in pieces:
            yield StreamChunk(text=piece)
        yield StreamChunk(text="", done=True)
