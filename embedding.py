"""SRAG Embedding — Phase 1 stub (SD.md §3, DD-03).

Provides the Ollama embedding client wrapper. Currently a stub;
real implementation is a Phase 2 concern (DD-03).
"""

from __future__ import annotations


class Embedder:
    """Ollama embedding client stub.

    Will be wired to `ollama.embed` in Phase 2 per DD-03.
    """

    def __init__(self, url: str = "http://localhost:11434", model: str = "qwen3-embedding") -> None:
        self._url = url
        self._model = model

    def embed(self, text: str) -> list[float]:
        """Embed a single text. Returns 4096-dim vector."""
        # TODO: Phase 2 - call ollama.embed(model=self._model, input=text)
        return [0.0] * 4096

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts. Returns one 4096-dim vector per text."""
        # TODO: Phase 2 - call ollama.embed(model=self._model, input=texts) in batch mode
        return [self.embed(t) for t in texts]


__all__ = ["Embedder"]