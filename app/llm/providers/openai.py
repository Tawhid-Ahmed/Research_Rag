"""OpenAI chat provider with native async streaming."""

from __future__ import annotations

from collections.abc import AsyncIterator

from app.llm.types import ChatMessage, GenerationConfig, StreamChunk


class OpenAIProvider:
    """Chat provider backed by the OpenAI Chat Completions API."""

    def __init__(
        self,
        *,
        model_name: str,
        api_key: str | None,
        temperature: float,
        max_tokens: int,
    ) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client: object | None = None

    def _get_client(self) -> object:
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=self.api_key)
        return self._client

    @staticmethod
    def _to_openai_messages(messages: list[ChatMessage]) -> list[dict[str, str]]:
        return [{"role": msg.role, "content": msg.content} for msg in messages]

    async def generate(self, messages: list[ChatMessage], config: GenerationConfig) -> str:
        client = self._get_client()
        response = await client.chat.completions.create(  # type: ignore[attr-defined]
            model=self.model_name,
            messages=self._to_openai_messages(messages),
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        content = response.choices[0].message.content
        return content or ""

    async def stream_generate(
        self, messages: list[ChatMessage], config: GenerationConfig
    ) -> AsyncIterator[StreamChunk]:
        client = self._get_client()
        stream = await client.chat.completions.create(  # type: ignore[attr-defined]
            model=self.model_name,
            messages=self._to_openai_messages(messages),
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            stream=True,
        )
        async for event in stream:
            delta = event.choices[0].delta.content if event.choices else None
            if delta:
                yield StreamChunk(text=delta)
        yield StreamChunk(text="", done=True)
