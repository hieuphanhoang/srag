"""SRAG Embedding — Phase 2 implementation (DD-03).

Provides the Ollama embedding client wrapper with batch embedding,
dimension inference, and configurable models.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Lazy ollama import to allow startup without ollama running.
_ollama: Any | None = None


def _get_ollama() -> Any:
    """Return the ollama module (imports once)."""
    global _ollama
    if _ollama is None:
        try:
            import ollama as _mod
            _ollama = _mod
        except ImportError as exc:
            raise RuntimeError(
                "The 'ollama' package is required for embedding. "
                "Install it with: pip install ollama"
            ) from exc
    return _ollama


class Embedder:
    """Ollama embedding client.

    Parameters
    ----------
    url : str
        Ollama API base URL.
    model : str
        Embedding model name (default *nomic-embed-base*).
    dimension : int | None
        Target dimension; ``None`` means infer from the model response.
    """

    def __init__(
        self,
        url: str = "http://localhost:11434",
        model: str = "qwen3-embedding",
        dimension: int | None = None,
    ) -> None:
        self._url = url
        self._model = model
        self._dimension = dimension
        self._client: Any | None = None

    def _get_client(self) -> Any:
        """Return (building on first use) a Client bound to self._url.

        `ollama.embed(...)` is a module-level convenience function that
        always targets ollama's default global client - it has no host/url
        parameter at all - so calling it here would silently ignore
        `self._url` and always hit the default Ollama endpoint regardless
        of what was configured. Building our own Client(host=self._url)
        (matching llm/ollama_provider.py's OllamaProvider) is what actually
        makes a custom Ollama URL take effect.
        """
        if self._client is None:
            ollama = _get_ollama()
            self._client = ollama.Client(host=self._url)
        return self._client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, text: str) -> list[float]:
        """Embed a single text string.

        Returns
        -------
        list[float]
            Normalised embedding vector (L2-normalised if the model supports it).
        """
        client = self._get_client()
        resp = client.embed(model=self._model, input=text, options={"num_batch": 512})
        emb = resp["embeddings"]
        # ollama returns a list-of-lists even for single input.
        vec = emb[0] if isinstance(emb[0], list) else [float(emb[0])]  # type: ignore[arg-type]

        if self._dimension and len(vec) > self._dimension:
            vec = vec[: self._dimension]
        return vec

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple text strings in a single Ollama call.

        Parameters
        ----------
        texts : list[str]
            Texts to embed (max 512 per batch for most models).

        Returns
        -------
        list[list[float]]
            One normalised vector per input text.
        """
        if not texts:
            return []

        client = self._get_client()
        resp = client.embed(model=self._model, input=texts, options={"num_batch": 512})
        embeddings = resp["embeddings"]

        results: list[list[float]] = []
        for emb in embeddings:
            vec = emb if isinstance(emb, list) else [float(emb)]  # type: ignore[arg-type]
            if self._dimension and len(vec) > self._dimension:
                vec = vec[: self._dimension]
            results.append(vec)
        return results

    @property
    def dimension(self) -> int:
        """Return the embedding dimension (inferred on first call)."""
        if self._dimension is None:
            # Infer from a dummy embed.
            dummy = self.embed("\x00")  # minimal input
            self._dimension = len(dummy)
            logger.info("Inferred embedding dimension: %d", self._dimension)
        return self._dimension


# Convenience aliases for backward compatibility / cross-module imports.
OllamaEmbeddingModel = Embedder  # type: ignore[misc]
HuggingFaceInferenceModel = Embedder  # type: ignore[misc]

# Re-export key symbols so other modules can import them.
EmbeddingModel = Embedder


def get_embedder(url: str | None = None, model: str | None = None) -> "Embedder":
    """Return a cached *Embedder* instance (singleton).

    Parameters follow :class:`Embedder.__init__`.
    """
    import os  # noqa: local import for lazy loading
    from config import load_config

    cfg = load_config()
    _url = url or os.environ.get("OLLAMA_URL") or cfg.ollama_url
    _model = model or cfg.embedding_model
    return Embedder(url=_url, model=_model)


__all__ = [
    "Embedder",
    "EmbeddingModel",
    "OllamaEmbeddingModel",
    "HuggingFaceInferenceModel",
    "get_embedder",
]
