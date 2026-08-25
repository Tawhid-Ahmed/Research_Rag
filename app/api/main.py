"""FastAPI application: health, ingest, and RAG query endpoints."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI, HTTPException
from sse_starlette.sse import EventSourceResponse

from app import __version__
from app.api.deps import IngestPipelineDep, LLMDep, ObservabilityDep, RetrieverDep, SettingsDep
from app.api.rag import (
    REFUSAL_MESSAGE,
    build_rag_messages,
    confidence_from_hits,
    hits_to_citations,
    should_refuse,
    summarize_retrieval,
)
from app.api.schemas import IngestRequest, IngestResponse, QueryRequest, QueryResponse, TokenUsage
from app.api.streaming import iter_query_sse
from app.config import get_settings
from app.llm.types import GenerationConfig, StreamChunk
from app.observability.tracing import QueryTrace, compute_usage

logger = logging.getLogger(__name__)

settings = get_settings()
app = FastAPI(title=settings.app_name, version=__version__)


def _usage_model(usage: Any) -> TokenUsage:
    return TokenUsage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
        estimated_cost_usd=usage.estimated_cost_usd,
    )


@app.get("/health")
def health(obs: ObservabilityDep) -> dict[str, str]:
    """Liveness/readiness probe used by Docker, CI, and HF Spaces."""

    cfg = get_settings()
    return {
        "status": "ok",
        "app": cfg.app_name,
        "version": __version__,
        "environment": cfg.environment,
        "langfuse": "enabled" if obs.enabled else "disabled",
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
    obs: ObservabilityDep,
) -> QueryResponse | EventSourceResponse:
    """Answer a question over the indexed corpus with citations.

    When ``stream=true``, returns Server-Sent Events:
    ``meta`` → ``token``* → ``[DONE]``.
    """

    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="question must not be empty")

    provider_name = settings.llm_provider.value
    model_name = settings.llm_model
    trace = obs.start_query(
        question,
        provider=provider_name,
        model=model_name,
        metadata={"stream": body.stream},
    )

    result = await asyncio.to_thread(retriever.search, question)
    hits = result.hits
    if body.top_k is not None:
        hits = hits[: body.top_k]

    confidence = confidence_from_hits(hits)
    citations = hits_to_citations(hits)
    refused = should_refuse(confidence, settings.min_confidence_score) or not hits
    retrieval_summary = summarize_retrieval(result)

    trace.record_retrieval(
        retrieval=retrieval_summary,
        confidence=confidence,
        refused=refused,
        citation_count=len(citations),
    )

    gen_config = GenerationConfig(
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )

    if refused:
        answer = REFUSAL_MESSAGE
        usage = compute_usage(
            messages=[],
            answer=answer,
            provider=provider_name,
            model=model_name,
        )
        trace.end(
            output=answer,
            metadata={
                "refused": True,
                "confidence": confidence,
                "usage": usage.as_dict(),
            },
        )
        if body.stream:
            return _stream_refused(
                question=question,
                answer=answer,
                citations=citations,
                confidence=confidence,
                provider=provider_name,
                model=model_name,
                retrieval=retrieval_summary,
                usage=_usage_model(usage),
            )
        return QueryResponse(
            question=question,
            answer=answer,
            citations=citations,
            refused=True,
            confidence=confidence,
            provider=provider_name,
            model=model_name,
            usage=_usage_model(usage),
        )

    messages = build_rag_messages(question, hits)

    if body.stream:
        meta = {
            "question": question,
            "citations": [c.model_dump() for c in citations],
            "refused": False,
            "confidence": confidence,
            "provider": provider_name,
            "model": model_name,
            "retrieval": retrieval_summary,
        }
        return EventSourceResponse(
            _traced_token_stream(
                llm=llm,
                messages=messages,
                gen_config=gen_config,
                meta=meta,
                trace=trace,
                provider=provider_name,
                model=model_name,
            )
        )

    answer = await llm.generate(messages, gen_config)
    usage = compute_usage(
        messages=messages,
        answer=answer,
        provider=provider_name,
        model=model_name,
    )
    trace.record_generation(
        messages=messages,
        answer=answer,
        provider=provider_name,
        model=model_name,
        usage=usage,
    )
    trace.end(
        output=answer,
        metadata={"refused": False, "confidence": confidence, "usage": usage.as_dict()},
    )
    return QueryResponse(
        question=question,
        answer=answer,
        citations=citations,
        refused=False,
        confidence=confidence,
        provider=provider_name,
        model=model_name,
        usage=_usage_model(usage),
    )


async def _traced_token_stream(
    *,
    llm: Any,
    messages: list,
    gen_config: GenerationConfig,
    meta: dict[str, Any],
    trace: QueryTrace,
    provider: str,
    model: str,
) -> AsyncIterator[dict[str, str]]:
    """Stream tokens, then emit usage and finalize Langfuse after completion."""

    import json

    from app.llm.sse import DONE_SENTINEL

    answer_parts: list[str] = []

    async def token_source() -> AsyncIterator[StreamChunk]:
        async for chunk in llm.stream_generate(messages, gen_config):
            if chunk.text:
                answer_parts.append(chunk.text)
            yield chunk

    async for frame in iter_query_sse(meta=meta, token_stream=token_source()):
        if frame.get("data") == DONE_SENTINEL:
            answer = "".join(answer_parts)
            usage = compute_usage(
                messages=messages,
                answer=answer,
                provider=provider,
                model=model,
            )
            trace.record_generation(
                messages=messages,
                answer=answer,
                provider=provider,
                model=model,
                usage=usage,
            )
            trace.end(
                output=answer,
                metadata={
                    "refused": False,
                    "confidence": meta.get("confidence"),
                    "usage": usage.as_dict(),
                    "stream": True,
                },
            )
            yield {
                "data": json.dumps(
                    {"type": "usage", **_usage_model(usage).model_dump()},
                    ensure_ascii=False,
                )
            }
        yield frame


def _stream_refused(
    *,
    question: str,
    answer: str,
    citations: list,
    confidence: float,
    provider: str,
    model: str,
    retrieval: dict[str, int],
    usage: TokenUsage,
) -> EventSourceResponse:
    """Stream a refusal as meta + one token frame + DONE."""

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
        "usage": usage.model_dump(),
    }
    return EventSourceResponse(iter_query_sse(meta=meta, token_stream=tokens()))
