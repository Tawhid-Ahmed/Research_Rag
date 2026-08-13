"""LLM provider interface.

This step focuses on wiring a *single* interface that:
1) can be implemented by multiple backends, and
2) supports streaming so Step 5 can stream responses over SSE.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from app.llm.types import ChatMessage, GenerationConfig, StreamChunk


class ChatProvider(Protocol):
    """Abstract chat provider (OpenAI/Anthropic/HF/Ollama)."""

    async def generate(self, messages: list[ChatMessage], config: GenerationConfig) -> str: ...

    def stream_generate(
        self, messages: list[ChatMessage], config: GenerationConfig
    ) -> AsyncIterator[StreamChunk]: ...
