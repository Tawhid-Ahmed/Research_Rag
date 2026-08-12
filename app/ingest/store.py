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

    def get_all(self) -> list[Chunk]:
        """Load every indexed chunk (text + metadata) for sparse indexing."""

        if self.count() == 0:
            return []
        raw = self.collection.get(include=["documents", "metadatas"])  # type: ignore[attr-defined]
        return _rows_to_chunks(raw["ids"], raw["documents"], raw["metadatas"])

    def query_by_embedding(
        self,
        embedding: list[float],
        top_k: int,
    ) -> list[tuple[Chunk, float]]:
        """Nearest-neighbor search; returns ``(chunk, cosine_similarity)`` pairs.

        Chroma stores cosine *distance* (``1 - similarity``) when the collection
        was created with ``hnsw:space=cosine``. We convert back to similarity so
        callers can treat higher scores as better, matching BM25.
        """

        n = self.count()
        if n == 0 or top_k <= 0:
            return []
        raw = self.collection.query(  # type: ignore[attr-defined]
            query_embeddings=[embedding],
            n_results=min(top_k, n),
            include=["documents", "metadatas", "distances"],
        )
        ids = raw["ids"][0]
        documents = raw["documents"][0]
        metadatas = raw["metadatas"][0]
        distances = raw["distances"][0]
        chunks = _rows_to_chunks(ids, documents, metadatas)
        hits: list[tuple[Chunk, float]] = []
        for chunk, distance in zip(chunks, distances, strict=True):
            hits.append((chunk, 1.0 - float(distance)))
        return hits

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


def _rows_to_chunks(
    ids: list[str],
    documents: list[str] | None,
    metadatas: list[dict[str, object] | None] | None,
) -> list[Chunk]:
    """Rebuild :class:`Chunk` objects from a Chroma ``get``/``query`` payload."""

    docs = documents or [""] * len(ids)
    metas = metadatas or [None] * len(ids)
    chunks: list[Chunk] = []
    for chunk_id, text, meta in zip(ids, docs, metas, strict=True):
        data = meta or {}
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                arxiv_id=str(data.get("arxiv_id", "")),
                title=str(data.get("title", "")),
                text=text or "",
                token_count=_as_int(data.get("token_count")),
                chunk_index=_as_int(data.get("chunk_index")),
                page_start=_as_int(data.get("page_start")),
                page_end=_as_int(data.get("page_end")),
                source_url=str(data.get("source_url", "")),
            )
        )
    return chunks


def _as_int(value: object, default: int = 0) -> int:
    """Coerce Chroma metadata (typed as ``object``) to int."""

    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default
