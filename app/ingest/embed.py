"""Dense embeddings via ``sentence-transformers``.

The model is loaded lazily on first use (it can be hundreds of MB and downloads
weights on first run), so importing this module -- or constructing the embedder
in tests -- stays cheap. The same embedder is used at ingest time (documents)
and, later, at query time, guaranteeing the vectors live in the same space.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid importing the heavy dependency at module load
    from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbedder:
    """Wraps a ``sentence-transformers`` model with batched encoding."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str | None = None,
        batch_size: int = 32,
        normalize: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Load and cache the underlying model on first access."""

        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents into a list of float vectors."""

        if not texts:
            return []
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string (mirrors :meth:`embed_texts`)."""

        return self.embed_texts([text])[0]
