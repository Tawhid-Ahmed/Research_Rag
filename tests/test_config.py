"""Smoke tests for the configuration layer.

These give CI something green to run from the very first commit and verify the
scaffold imports cleanly.
"""

from __future__ import annotations

from app.config import LLMProvider, Settings, VectorStore, get_settings


def test_settings_defaults() -> None:
    settings = Settings()
    assert settings.app_name
    assert settings.api_port == 8000
    assert settings.chunk_overlap < settings.chunk_size
    assert settings.rerank_top_k <= settings.retrieval_top_k


def test_default_provider_and_store_are_free_path() -> None:
    settings = Settings()
    assert settings.llm_provider is LLMProvider.HUGGINGFACE
    assert settings.vector_store is VectorStore.CHROMA


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()


def test_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("API_PORT", "9001")
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.llm_provider is LLMProvider.OLLAMA
    assert settings.api_port == 9001
    get_settings.cache_clear()
