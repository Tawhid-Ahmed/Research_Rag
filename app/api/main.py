"""FastAPI application entrypoint.

Scaffold only: exposes a minimal ``/health`` endpoint so the container, CI, and
local dev have a working ASGI app from day one. Query/ingest routes are added in
later build steps.
"""

from __future__ import annotations

from fastapi import FastAPI

from app import __version__
from app.config import get_settings

settings = get_settings()

app = FastAPI(title=settings.app_name, version=__version__)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness/readiness probe used by Docker, CI, and HF Spaces."""

    return {
        "status": "ok",
        "app": settings.app_name,
        "version": __version__,
        "environment": settings.environment,
    }
