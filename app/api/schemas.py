"""HTTP request/response schemas for the FastAPI layer."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class IngestRequest(BaseModel):
    """Body for ``POST /ingest``."""

    query: str | None = None
    ids: list[str] | None = None
    max_results: int = Field(default=10, ge=1, le=50)

    @model_validator(mode="after")
    def require_query_or_ids(self) -> IngestRequest:
        if not self.query and not self.ids:
            raise ValueError("provide either query or ids")
        return self


class IngestResponse(BaseModel):
    """Summary returned after an ingestion run."""

    papers_fetched: int
    papers_indexed: int
    pages_parsed: int
    chunks_indexed: int
    collection_count: int
    skipped: list[str] = Field(default_factory=list)


class QueryRequest(BaseModel):
    """Body for ``POST /query``."""

    question: str = Field(min_length=1)
    stream: bool = False
    top_k: int | None = Field(default=None, ge=1, le=20)


class Citation(BaseModel):
    """One retrieved chunk cited in the answer."""

    chunk_id: str
    arxiv_id: str
    title: str
    page_start: int
    page_end: int
    score: float
    source_url: str = ""
    excerpt: str = ""


class QueryResponse(BaseModel):
    """Non-streaming query result."""

    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    refused: bool = False
    confidence: float = 0.0
    provider: str = ""
    model: str = ""
