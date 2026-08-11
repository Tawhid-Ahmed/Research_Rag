"""Persistent vector index backed by Chroma.

We pass embeddings in explicitly (rather than letting Chroma embed for us) so
the exact same :class:`SentenceTransformerEmbedder` is the single source of
truth for vectors across ingest and query time. ``upsert`` keyed on the
deterministic ``chunk_id`` makes re-ingesting a paper idempotent.

``chromadb`` is imported lazily so this module is cheap to import and the client
(which opens an on-disk DB) is only created when actually used.
"""

from __future__ import annotations

from pathlib import Path

from app.ingest.models import Chunk


class ChromaVectorStore:
    """Thin wrapper over a persistent Chroma collection."""

    def __init__(self, persist_dir: Path, collection_name: str) -> None:
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self._collection: object | None = None

    @property
    def collection(self) -> object:
        """Open (or create) the persistent collection on first use."""

        if self._collection is None:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            self.persist_dir.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            # Cosine space matches normalized sentence-transformer vectors.
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Upsert chunks and their embeddings keyed by ``chunk_id``."""

        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must be the same length")
        self.collection.upsert(  # type: ignore[attr-defined]
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=embeddings,
            metadatas=[chunk.to_metadata() for chunk in chunks],
        )

    def count(self) -> int:
        """Return the number of indexed chunks."""

        return int(self.collection.count())  # type: ignore[attr-defined]

    def reset(self) -> None:
        """Delete and recreate the collection (drops all indexed data)."""

        import contextlib

        import chromadb
        from chromadb.config import Settings as ChromaSettings

        self.persist_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        # Collection may not exist yet; recreating below is enough.
        with contextlib.suppress(Exception):
            client.delete_collection(self.collection_name)
        self._collection = client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
