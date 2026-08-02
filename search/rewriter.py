"""SRAG Search Rewriter — Phase 1 stub (DD-05, FD-32…FD-34).

Query rewriting module. Currently a stub; real implementation is
a Phase 2 concern.
"""

from __future__ import annotations


class QueryRewriter:
    """Query rewriter stub."""

    def __init__(self, llm_provider) -> None:  # type: ignore[no-def-no-attr]
        self._llm = llm_provider

    def rewrite(self, query: str) -> list[str]:
        """Rewrite a query for search optimization."""
        # TODO: Phase 2 - implement per FD-32…FD-34 and DD-05
        return [query]


__all__ = ["QueryRewriter"]