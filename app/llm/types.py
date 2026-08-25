"""Shared LLM data types.

The rest of the app (API/UI) should talk to providers via these contracts so
we can swap OpenAI/Anthropic/Hugging Face/Ollama without refactoring logic.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant"]


class ChatMessage(BaseModel):
    """One message in a chat conversation."""

    role: Role
    content: str


class GenerationConfig(BaseModel):
    """Generation knobs common across LLM providers."""

    temperature: float = Field(default=0.1, ge=0.0)
    max_tokens: int = Field(default=1024, ge=1)


class StreamChunk(BaseModel):
    """One streaming payload unit (sent to API/SSE later).

    ``done=True`` marks the end of the stream so SSE clients can close cleanly.
    """

    text: str = ""
    done: bool = False
