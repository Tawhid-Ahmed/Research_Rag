"""FastAPI dependency providers (overridable in tests)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated

from fastapi import Depends

from app.config import Settings, get_settings
from app.ingest.pipeline import IngestionPipeline, build_default_pipeline
from app.llm.factory import build_provider
from app.llm.interfaces import ChatProvider
from app.retrieval.pipeline import RetrievalPipeline, build_default_retriever

SettingsDep = Annotated[Settings, Depends(get_settings)]

# Module-level factories so tests can monkeypatch without touching routes.
_ingest_factory: Callable[[Settings], IngestionPipeline] = build_default_pipeline
_retriever_factory: Callable[[Settings], RetrievalPipeline] = build_default_retriever
_llm_factory: Callable[[Settings], ChatProvider] = build_provider


def get_ingest_pipeline(settings: SettingsDep) -> IngestionPipeline:
    return _ingest_factory(settings)


def get_retriever(settings: SettingsDep) -> RetrievalPipeline:
    return _retriever_factory(settings)


def get_llm(settings: SettingsDep) -> ChatProvider:
    return _llm_factory(settings)


IngestPipelineDep = Annotated[IngestionPipeline, Depends(get_ingest_pipeline)]
RetrieverDep = Annotated[RetrievalPipeline, Depends(get_retriever)]
LLMDep = Annotated[ChatProvider, Depends(get_llm)]
