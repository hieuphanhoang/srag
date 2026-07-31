"""SRAG Search Merge — Phase 1 stub (DD-05).

Result deduplication and merge module. Currently a stub; real implementation
is a Phase 2 concern.
"""

from __future__ import annotations


def merge_results(original: list[dict], rewritten: list[dict]) -> list[dict]:
    """Merge and deduplicate search results from original and rewritten queries."""
    # TODO: Phase 2 - implement per DD-05
    return original + rewritten


__all__ = ["merge_results"]