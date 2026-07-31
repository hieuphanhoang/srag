"""SRAG LLM Base — Phase 1 stub (DD-07, FD-29…FD-31).

LLM provider protocol base class. Currently a stub; real implementation
is a Phase 2 concern.
"""

from __future__ import annotations

from typing import Protocol


class LLMProvider(Protocol):
    """Abstract protocol for an LLM provider."""

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate text from a prompt."""
        ...


__all__ = ["LLMProvider"]