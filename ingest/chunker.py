"""SRAG Ingest Chunker — Phase 1 stub (DD-04, FD-26…FD-28).

Text → token-based chunks. Currently a stub; real implementation is
a Phase 2 concern.
"""

from __future__ import annotations


class TextChunker:
    """Token-based text chunker stub."""

    def __init__(self, max_tokens: int = 768, overlap: int = 64) -> None:
        self._max_tokens = max_tokens
        self._overlap = overlap

    def chunk(self, text: str) -> list[str]:
        """Split text into chunks of ~max_tokens tokens."""
        # TODO: Phase 2 - implement per FD-26…FD-28 using tiktoken
        return [text]


__all__ = ["TextChunker"]