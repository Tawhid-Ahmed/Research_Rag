"""Anthropic Claude provider with native async streaming."""

from __future__ import annotations

from collections.abc import AsyncIterator

from app.llm.providers.base import split_system_and_chat
from app.llm.types import ChatMessage, GenerationConfig, StreamChunk


class AnthropicProvider:
    """Chat provider backed by the Anthropic Messages API."""

    def __init__(
        self,
        *,
        model_name: str,
        api_key: str | None,
        temperature: float,
        max_tokens: int,
    ) -> None:
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic")
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client: object | None = None

    def _get_client(self) -> object:
        if self._client is None:
            from anthropic import AsyncAnthropic

            self._client = AsyncAnthropic(api_key=self.api_key)
        return self._client

    @staticmethod
    def _to_anthropic_messages(messages: list[ChatMessage]) -> list[dict[str, str]]:
        return [{"role": msg.role, "content": msg.content} for msg in messages]

    async def generate(self, messages: list[ChatMessage], config: GenerationConfig) -> str:
        client = self._get_client()
        system, chat = split_system_and_chat(messages)
        kwargs: dict[str, object] = {
            "model": self.model_name,
            "messages": self._to_anthropic_messages(chat),
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        }
        if system:
            kwargs["system"] = system
        response = await client.messages.create(**kwargs)  # type: ignore[attr-defined]
        parts = [block.text for block in response.content if getattr(block, "type", "") == "text"]
        return "".join(parts)

    async def stream_generate(
        self, messages: list[ChatMessage], config: GenerationConfig
    ) -> AsyncIterator[StreamChunk]:
        client = self._get_client()
        system, chat = split_system_and_chat(messages)
        kwargs: dict[str, object] = {
            "model": self.model_name,
            "messages": self._to_anthropic_messages(chat),
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        }
        if system:
            kwargs["system"] = system

        async with client.messages.stream(**kwargs) as stream:  # type: ignore[attr-defined]
            async for text in stream.text_stream:
                if text:
                    yield StreamChunk(text=text)
        yield StreamChunk(text="", done=True)
