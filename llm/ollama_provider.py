"""SRAG Ollama Provider — Phase 1 stub (DD-07).

Ollama chat completion provider. Currently a stub; real implementation
is a Phase 2 concern.
"""

from __future__ import annotations

from .base import LLMProvider


class OllamaLLMProvider(LLMProvider):
    """Ollama LLM provider stub."""

    def __init__(self, url: str = "http://localhost:11434", model: str = "qwen3:4b") -> None:
        self._url = url
        self._model = model

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate text from a prompt."""
        # TODO: Phase 2 - call Ollama API
        raise NotImplementedError("Ollama provider not implemented yet")


__all__ = ["OllamaLLMProvider"]