"""LLM provider abstraction (OpenAI / Anthropic / Hugging Face / Ollama) with streaming."""

from __future__ import annotations

from app.llm.factory import build_provider
from app.llm.interfaces import ChatProvider
from app.llm.sse import DONE_SENTINEL, encode_sse_chunk, iter_sse
from app.llm.types import ChatMessage, GenerationConfig, StreamChunk

__all__ = [
    "DONE_SENTINEL",
    "ChatMessage",
    "ChatProvider",
    "GenerationConfig",
    "StreamChunk",
    "build_provider",
    "encode_sse_chunk",
    "iter_sse",
]
