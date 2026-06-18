"""Application configuration.

Centralized, typed settings loaded from environment variables and an optional
``.env`` file via ``pydantic-settings``. Import the cached :func:`get_settings`
accessor anywhere in the app; never read ``os.environ`` directly.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root: ``.../arxiv-rag``. Used to resolve default data paths.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class LLMProvider(str, Enum):
    """Supported LLM backends, selected via the ``LLM_PROVIDER`` env var."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"


class VectorStore(str, Enum):
    """Supported vector stores, selected via the ``VECTOR_STORE`` env var."""

    CHROMA = "chroma"
    PGVECTOR = "pgvector"


class Settings(BaseSettings):
    """Typed application settings.

    Values are read (in order of precedence) from constructor args, environment
    variables, then the ``.env`` file. See ``.env.example`` for documentation of
    every field.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- App ---------------------------------------------------------------
    app_name: str = "arXiv RAG Assistant"
    environment: str = Field(default="development")
    log_level: str = Field(default="INFO")
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)

    # --- LLM provider ------------------------------------------------------
    llm_provider: LLMProvider = Field(default=LLMProvider.HUGGINGFACE)
    llm_model: str = Field(default="HuggingFaceH4/zephyr-7b-beta")
    llm_temperature: float = Field(default=0.1)
    llm_max_tokens: int = Field(default=1024)

    openai_api_key: str | None = Field(default=None)
    anthropic_api_key: str | None = Field(default=None)
    huggingface_api_key: str | None = Field(default=None)
    ollama_base_url: str = Field(default="http://localhost:11434")

    # --- Embeddings & retrieval -------------------------------------------
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=64)
    retrieval_top_k: int = Field(default=20)
    rerank_top_k: int = Field(default=5)
    # Below this reranker score the app refuses to answer (low-confidence guardrail).
    min_confidence_score: float = Field(default=0.2)

    # --- Vector store ------------------------------------------------------
    vector_store: VectorStore = Field(default=VectorStore.CHROMA)
    chroma_persist_dir: Path = Field(default=PROJECT_ROOT / "data" / "chroma")
    collection_name: str = Field(default="arxiv_papers")
    # Used only when ``vector_store == pgvector``.
    postgres_dsn: str | None = Field(default=None)

    # --- Data paths --------------------------------------------------------
    data_dir: Path = Field(default=PROJECT_ROOT / "data")
    pdf_cache_dir: Path = Field(default=PROJECT_ROOT / "data" / "pdfs")

    # --- Observability -----------------------------------------------------
    langfuse_public_key: str | None = Field(default=None)
    langfuse_secret_key: str | None = Field(default=None)
    langfuse_host: str = Field(default="https://cloud.langfuse.com")


@lru_cache
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance."""

    return Settings()
