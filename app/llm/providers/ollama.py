"""Local Ollama chat provider via the HTTP API."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from app.llm.types import ChatMessage, GenerationConfig, StreamChunk


class OllamaProvider:
    """Chat provider backed by a local Ollama server (``/api/chat``)."""

    def __init__(
        self,
        *,
        model_name: str,
        base_url: str,
        temperature: float,
        max_tokens: int,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens

    @staticmethod
    def _to_ollama_messages(messages: list[ChatMessage]) -> list[dict[str, str]]:
        return [{"role": msg.role, "content": msg.content} for msg in messages]

    def _payload(
        self, messages: list[ChatMessage], config: GenerationConfig, *, stream: bool
    ) -> dict:
        return {
            "model": self.model_name,
            "messages": self._to_ollama_messages(messages),
            "stream": stream,
            "options": {
                "temperature": config.temperature,
                "num_predict": config.max_tokens,
            },
        }

    async def generate(self, messages: list[ChatMessage], config: GenerationConfig) -> str:
        import httpx

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json=self._payload(messages, config, stream=False),
            )
            response.raise_for_status()
            data = response.json()
            return str(data.get("message", {}).get("content", ""))

    async def stream_generate(
        self, messages: list[ChatMessage], config: GenerationConfig
    ) -> AsyncIterator[StreamChunk]:
        import httpx

        async with (
            httpx.AsyncClient(timeout=120.0) as client,
            client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json=self._payload(messages, config, stream=True),
            ) as response,
        ):
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                data = json.loads(line)
                content = data.get("message", {}).get("content", "")
                if content:
                    yield StreamChunk(text=content)
                if data.get("done"):
                    break
        yield StreamChunk(text="", done=True)
