"""Unit tests for LLM provider selection from config."""

from __future__ import annotations

import pytest

from app.config import LLMProvider, Settings
from app.llm.factory import build_provider
from app.llm.providers.anthropic import AnthropicProvider
from app.llm.providers.huggingface import HuggingFaceProvider
from app.llm.providers.ollama import OllamaProvider
from app.llm.providers.openai import OpenAIProvider


def test_build_provider_defaults_to_huggingface() -> None:
    settings = Settings()
    assert settings.llm_provider is LLMProvider.HUGGINGFACE
    provider = build_provider(settings)
    assert isinstance(provider, HuggingFaceProvider)


def test_build_provider_openai_requires_key() -> None:
    settings = Settings(llm_provider=LLMProvider.OPENAI, openai_api_key=None)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        build_provider(settings)


def test_build_provider_openai() -> None:
    settings = Settings(
        llm_provider=LLMProvider.OPENAI,
        openai_api_key="sk-test",
        llm_model="gpt-4o-mini",
    )
    provider = build_provider(settings)
    assert isinstance(provider, OpenAIProvider)
    assert provider.model_name == "gpt-4o-mini"


def test_build_provider_anthropic() -> None:
    settings = Settings(
        llm_provider=LLMProvider.ANTHROPIC,
        anthropic_api_key="sk-ant-test",
        llm_model="claude-3-5-haiku-latest",
    )
    provider = build_provider(settings)
    assert isinstance(provider, AnthropicProvider)


def test_build_provider_ollama() -> None:
    settings = Settings(
        llm_provider=LLMProvider.OLLAMA,
        llm_model="llama3.2",
        ollama_base_url="http://localhost:11434",
    )
    provider = build_provider(settings)
    assert isinstance(provider, OllamaProvider)
    assert provider.base_url == "http://localhost:11434"


def test_env_selects_provider(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_MODEL", "mistral")
    from app.config import get_settings

    get_settings.cache_clear()
    settings = get_settings()
    provider = build_provider(settings)
    assert isinstance(provider, OllamaProvider)
    assert provider.model_name == "mistral"
    get_settings.cache_clear()
