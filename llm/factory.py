"""SRAG LLM Factory — Phase 1 stub (DD-07).

Factory for creating LLM provider instances. Currently a stub;
real implementation is a Phase 2 concern.
"""

from __future__ import annotations

from .base import LLMProvider


def create_llm_provider(provider: str, model: str | None = None) -> LLMProvider:
    """Create an LLM provider instance from 'provider/model' string."""
    # TODO: Phase 2 - implement per DD-07
    raise NotImplementedError("LLM factory not implemented yet")


__all__ = ["create_llm_provider"]