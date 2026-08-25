"""Optional Langfuse tracing for RAG queries.

When Langfuse keys are missing, :class:`NoOpObservability` records nothing so
local/CI runs stay offline. When keys are present, each ``/query`` becomes a
trace with a retrieval span and an LLM generation (usage + estimated cost).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

from app.config import Settings
from app.llm.types import ChatMessage
from app.observability.cost import estimate_cost_usd
from app.observability.tokens import count_message_tokens, count_text_tokens

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UsageStats:
    """Token counts and estimated USD for one LLM call."""

    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def as_dict(self) -> dict[str, int | float]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
        }


def compute_usage(
    *,
    messages: list[ChatMessage] | None,
    answer: str,
    provider: str,
    model: str,
) -> UsageStats:
    """Count prompt/completion tokens and estimate cost."""

    input_tokens = count_message_tokens(messages or [])
    output_tokens = count_text_tokens(answer)
    return UsageStats(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost_usd=estimate_cost_usd(
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ),
    )


class QueryTrace(Protocol):
    def record_retrieval(
        self,
        *,
        retrieval: dict[str, int],
        confidence: float,
        refused: bool,
        citation_count: int,
    ) -> None: ...

    def record_generation(
        self,
        *,
        messages: list[ChatMessage],
        answer: str,
        provider: str,
        model: str,
        usage: UsageStats,
    ) -> None: ...

    def end(self, *, output: str, metadata: dict[str, Any] | None = None) -> None: ...


class Observability(Protocol):
    @property
    def enabled(self) -> bool: ...

    def start_query(
        self,
        question: str,
        *,
        provider: str,
        model: str,
        metadata: dict[str, Any] | None = None,
    ) -> QueryTrace: ...


class NoOpTrace:
    """Discard all observation events."""

    def record_retrieval(
        self,
        *,
        retrieval: dict[str, int],
        confidence: float,
        refused: bool,
        citation_count: int,
    ) -> None:
        return None

    def record_generation(
        self,
        *,
        messages: list[ChatMessage],
        answer: str,
        provider: str,
        model: str,
        usage: UsageStats,
    ) -> None:
        return None

    def end(self, *, output: str, metadata: dict[str, Any] | None = None) -> None:
        return None


class NoOpObservability:
    """Used when Langfuse is not configured."""

    @property
    def enabled(self) -> bool:
        return False

    def start_query(
        self,
        question: str,
        *,
        provider: str,
        model: str,
        metadata: dict[str, Any] | None = None,
    ) -> QueryTrace:
        return NoOpTrace()


class LangfuseQueryTrace:
    """One Langfuse trace spanning retrieval + generation for a query."""

    def __init__(self, client: Any, trace: Any) -> None:
        self._client = client
        self._trace = trace
        self._ended = False

    def record_retrieval(
        self,
        *,
        retrieval: dict[str, int],
        confidence: float,
        refused: bool,
        citation_count: int,
    ) -> None:
        self._trace.span(
            name="retrieval",
            input={
                "corpus_size": retrieval.get("corpus_size"),
                "top_k_pool": retrieval.get("retrieved"),
            },
            output={
                "reranked": retrieval.get("reranked"),
                "citation_count": citation_count,
                "confidence": confidence,
                "refused": refused,
            },
            metadata=retrieval,
        )

    def record_generation(
        self,
        *,
        messages: list[ChatMessage],
        answer: str,
        provider: str,
        model: str,
        usage: UsageStats,
    ) -> None:
        from langfuse.model import ModelUsage

        self._trace.generation(
            name="llm_generate",
            model=model,
            model_parameters={"provider": provider},
            input=[{"role": m.role, "content": m.content} for m in messages],
            output=answer,
            usage=ModelUsage(
                input=usage.input_tokens,
                output=usage.output_tokens,
                total=usage.total_tokens,
                unit="TOKENS",
            ),
            cost_details={"total": usage.estimated_cost_usd},
            metadata={
                "provider": provider,
                "estimated_cost_usd": usage.estimated_cost_usd,
            },
        )

    def end(self, *, output: str, metadata: dict[str, Any] | None = None) -> None:
        if self._ended:
            return
        self._ended = True
        self._trace.update(output=output, metadata=metadata or {})
        try:
            self._client.flush()
            trace_id = getattr(self._trace, "id", None)
            if trace_id:
                logger.info(
                    "langfuse trace flushed id=%s env=check-UI-filter-default-or-development",
                    trace_id,
                )
        except Exception:  # noqa: BLE001 - never break the request on telemetry
            logger.exception("langfuse flush failed")


class LangfuseObservability:
    """Langfuse-backed observability."""

    def __init__(self, client: Any, *, environment: str = "default") -> None:
        self._client = client
        self._environment = environment

    @property
    def enabled(self) -> bool:
        return True

    def start_query(
        self,
        question: str,
        *,
        provider: str,
        model: str,
        metadata: dict[str, Any] | None = None,
    ) -> QueryTrace:
        meta = {
            "provider": provider,
            "model": model,
            "environment": self._environment,
            **(metadata or {}),
        }
        trace = self._client.trace(
            name="rag_query",
            input={"question": question},
            metadata=meta,
            tags=["rag", provider, self._environment],
        )
        return LangfuseQueryTrace(self._client, trace)


def build_observability(settings: Settings | None = None) -> Observability:
    """Return Langfuse observability when keys are set; otherwise a no-op."""

    import os

    from app.config import get_settings

    settings = settings or get_settings()
    public_key = (settings.langfuse_public_key or "").strip()
    secret_key = (settings.langfuse_secret_key or "").strip()
    if not public_key or not secret_key:
        return NoOpObservability()

    # Langfuse UI filters by environment; align with app ENVIRONMENT (default: development).
    environment = (settings.environment or "development").strip() or "development"
    os.environ.setdefault("LANGFUSE_TRACING_ENVIRONMENT", environment)

    try:
        from langfuse import Langfuse

        client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=settings.langfuse_host,
            flush_at=1,
            flush_interval=0.5,
        )
        return LangfuseObservability(client, environment=environment)
    except Exception:  # noqa: BLE001 - missing/broken SDK should not stop the API
        logger.exception("failed to initialize Langfuse; continuing without tracing")
        return NoOpObservability()
