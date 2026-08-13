"""FastAPI application: health, ingest, and RAG query endpoints."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

from fastapi import FastAPI, HTTPException
from sse_starlette.sse import EventSourceResponse

from app import __version__
from app.api.deps import IngestPipelineDep, LLMDep, RetrieverDep, SettingsDep
from app.api.rag import (
    REFUSAL_MESSAGE,
    build_rag_messages,
    confidence_from_hits,
    hits_to_citations,
    should_refuse,
    summarize_retrieval,
)
from app.api.schemas import IngestRequest, IngestResponse, QueryRequest, QueryResponse
from app.api.streaming import iter_query_sse
from app.config import get_settings
from app.llm.types import GenerationConfig

logger = logging.getLogger(__name__)

settings = get_settings()
app = FastAPI(title=settings.app_name, version=__version__)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness/readiness probe used by Docker, CI, and HF Spaces."""

    cfg = get_settings()
    return {
        "status": "ok",
        "app": cfg.app_name,
        "version": __version__,
        "environment": cfg.environment,
    }


@app.post("/ingest", response_model=IngestResponse)
async def ingest(body: IngestRequest, pipeline: IngestPipelineDep) -> IngestResponse:
    """Fetch, chunk, embed, and index arXiv papers."""

    try:
        report = await asyncio.to_thread(
            pipeline.run,
            query=body.query,
            ids=body.ids,
            max_results=body.max_results,
        )
    except Exception as exc:  # noqa: BLE001 - surface upstream errors cleanly
        logger.exception("ingest failed")
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return IngestResponse(
        papers_fetched=report.papers_fetched,
        papers_indexed=report.papers_indexed,
        pages_parsed=report.pages_parsed,
        chunks_indexed=report.chunks_indexed,
        collection_count=report.collection_count,
        skipped=report.skipped,
    )


@app.post("/query", response_model=None)
async def query(
    body: QueryRequest,
    settings: SettingsDep,
    retriever: RetrieverDep,
    llm: LLMDep,
) -> QueryResponse | EventSourceResponse:
    """Answer a question over the indexed corpus with citations.

    When ``stream=true``, returns Server-Sent Events:
    ``meta`` → ``token``* → ``[DONE]``.
    """

    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="question must not be empty")

    result = await asyncio.to_thread(retriever.search, question)
    hits = result.hits
    if body.top_k is not None:
        hits = hits[: body.top_k]

    confidence = confidence_from_hits(hits)
    citations = hits_to_citations(hits)
    refused = should_refuse(confidence, settings.min_confidence_score) or not hits

    gen_config = GenerationConfig(
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )
    provider_name = settings.llm_provider.value
    model_name = settings.llm_model

    if refused:
        answer = REFUSAL_MESSAGE
        if body.stream:
            return _stream_refused(
                question=question,
                answer=answer,
                citations=citations,
                confidence=confidence,
                provider=provider_name,
                model=model_name,
                retrieval=summarize_retrieval(result),
            )
        return QueryResponse(
            question=question,
            answer=answer,
            citations=citations,
            refused=True,
            confidence=confidence,
            provider=provider_name,
            model=model_name,
        )

    messages = build_rag_messages(question, hits)

    if body.stream:
        from app.llm.types import StreamChunk

        async def token_source() -> AsyncIterator[StreamChunk]:
            async for chunk in llm.stream_generate(messages, gen_config):
                yield chunk

        meta = {
            "question": question,
            "citations": [c.model_dump() for c in citations],
            "refused": False,
            "confidence": confidence,
            "provider": provider_name,
            "model": model_name,
            "retrieval": summarize_retrieval(result),
        }
        return EventSourceResponse(iter_query_sse(meta=meta, token_stream=token_source()))

    answer = await llm.generate(messages, gen_config)
    return QueryResponse(
        question=question,
        answer=answer,
        citations=citations,
        refused=False,
        confidence=confidence,
        provider=provider_name,
        model=model_name,
    )


def _stream_refused(
    *,
    question: str,
    answer: str,
    citations: list,
    confidence: float,
    provider: str,
    model: str,
    retrieval: dict[str, int],
) -> EventSourceResponse:
    """Stream a refusal as meta + one token frame + DONE."""

    from app.llm.types import StreamChunk

    async def tokens() -> AsyncIterator[StreamChunk]:
        yield StreamChunk(text=answer)
        yield StreamChunk(done=True)

    meta = {
        "question": question,
        "citations": [c.model_dump() for c in citations],
        "refused": True,
        "confidence": confidence,
        "provider": provider,
        "model": model,
        "retrieval": retrieval,
    }
    return EventSourceResponse(iter_query_sse(meta=meta, token_stream=tokens()))
