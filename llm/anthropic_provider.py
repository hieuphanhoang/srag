"""SRAG Anthropic Provider — Phase 1 stub (DD-07).

Anthropic LLM provider stub. Currently a stub; real implementation
is a Phase 2 concern.
"""

from __future__ import annotations

from .base import LLMProvider


class AnthropicLLMProvider(LLMProvider):
    """Anthropic LLM provider stub."""

    def __init__(self, api_key: str | None = None, model: str = "claude-3-5-haiku-20241022") -> None:
        self._api_key = api_key
        self._model = model

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate text from a prompt."""
        # TODO: Phase 2 - call Anthropic Messages API
        raise NotImplementedError("Anthropic provider not implemented yet")


__all__ = ["AnthropicLLMProvider"]