"""Provider factory.

Maps :class:`app.config.LLMProvider` → concrete chat providers based on
runtime :class:`~app.config.Settings`.
"""

from __future__ import annotations

from app.config import LLMProvider, Settings, get_settings
from app.llm.interfaces import ChatProvider
from app.llm.providers.huggingface import HuggingFaceProvider


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

    raise NotImplementedError(f"LLM provider not implemented yet: {settings.llm_provider}")
