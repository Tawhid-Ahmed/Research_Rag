"""FastAPI application: health and ingest endpoints."""

from __future__ import annotations

import asyncio
import logging

from fastapi import FastAPI, HTTPException

from app import __version__
from app.api.deps import IngestPipelineDep
from app.api.schemas import IngestRequest, IngestResponse
from app.config import get_settings

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
