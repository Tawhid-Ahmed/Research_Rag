"""Provider factory.

Maps :class:`app.config.LLMProvider` → concrete chat providers based on
runtime :class:`~app.config.Settings`.
"""

from __future__ import annotations

from app.config import LLMProvider, Settings, get_settings
from app.llm.interfaces import ChatProvider
from app.llm.providers.anthropic import AnthropicProvider
from app.llm.providers.huggingface import HuggingFaceProvider
from app.llm.providers.ollama import OllamaProvider
from app.llm.providers.openai import OpenAIProvider


def build_provider(settings: Settings | None = None) -> ChatProvider:
    """Instantiate the configured LLM provider."""

    settings = settings or get_settings()

    if settings.llm_provider is LLMProvider.HUGGINGFACE:
        return HuggingFaceProvider(
            model_name=settings.llm_model,
            api_key=settings.huggingface_api_key,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )

    if settings.llm_provider is LLMProvider.OPENAI:
        return OpenAIProvider(
            model_name=settings.llm_model,
            api_key=settings.openai_api_key,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )

    if settings.llm_provider is LLMProvider.ANTHROPIC:
        return AnthropicProvider(
            model_name=settings.llm_model,
            api_key=settings.anthropic_api_key,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )

    if settings.llm_provider is LLMProvider.OLLAMA:
        return OllamaProvider(
            model_name=settings.llm_model,
            base_url=settings.ollama_base_url,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )

    raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")
