"""Ingestion pipeline: arXiv fetch, PDF parse, token-aware chunking, embed, index.

Public surface for the pipeline. Concrete implementations live in submodules and
import their heavy third-party dependencies lazily, so importing this package is
cheap and safe in tests.
"""

from __future__ import annotations

from app.ingest.chunk import ChunkSpan, TokenChunker, TokenEncoder
from app.ingest.embed import SentenceTransformerEmbedder
from app.ingest.fetch import ArxivFetcher
from app.ingest.models import Chunk, IngestionReport, Page, Paper
from app.ingest.parse import PdfParser, clean_text
from app.ingest.pipeline import IngestionPipeline, build_default_pipeline
from app.ingest.store import ChromaVectorStore

__all__ = [
    "ArxivFetcher",
    "Chunk",
    "ChromaVectorStore",
    "ChunkSpan",
    "IngestionPipeline",
    "IngestionReport",
    "Page",
    "Paper",
    "PdfParser",
    "SentenceTransformerEmbedder",
    "TokenChunker",
    "TokenEncoder",
    "build_default_pipeline",
    "clean_text",
]
