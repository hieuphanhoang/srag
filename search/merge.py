"""SRAG Search Merge — Result deduplication and scoring (DD-16, FD-54…FD-58).

Implements merging of search results from multiple query variants (original +
rewritten), with distance averaging for duplicates and global sorting.

Key design decisions:
- Deduplicates by (source, chunk_index) tuple — exact identity match.
- Averages distances across queries for the same chunk.
- Preserves source matches in SearchResult.sources for transparency.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

logger = logging.getLogger(__name__)


def merge_results(
    results: list[Any],
) -> list[Any]:
    """Merge and deduplicate search results from multiple queries.

    Groups duplicate chunks (by source + chunk_index), averages their distance
    scores, sorts globally by ascending average distance, and returns the result.

    Args:
        results: List of SearchResult objects or dicts from multiple query embeddings.
                Each should have `text`, `metadata` (with 'source' and 'chunk_index'),
                `distance`, and optionally `collection`.

    Returns:
        Merged list with duplicates removed, sorted by ascending average distance.
        Always returns all unique chunks — caller applies top-K limit.
    """
    if not results:
        return []

    # Step 0: Normalize to dict representation for consistent processing
    normalized = [_result_to_dict(r) for r in results]

    # Step 1: Build dedup map by (source, chunk_index)
    chunk_groups: dict[tuple[str | None, int | None], list[dict[str, Any]]] = {}
    for result in normalized:
        metadata = result.get("metadata", {}) or {}
        key = (metadata.get("source"), metadata.get("chunk_index"))

        if key not in chunk_groups:
            chunk_groups[key] = []
        chunk_groups[key].append(result)

    # Step 2: Merge duplicate groups via distance averaging
    merged: list[dict[str, Any]] = []
    for key, group in chunk_groups.items():
        if len(group) == 1:
            # No duplicate; keep as-is
            merged.append(group[0])
        else:
            # Multiple results for the same chunk — average distances
            avg_distance = sum(r.get("distance", 0.0) or 0.0 for r in group) / len(group)

            # Keep first result's metadata, update distance, store sources
            merged_result = dict(group[0])  # shallow copy of the first match
            merged_result["distance"] = avg_distance
            merged_result["_sources"] = group  # preserve source matches for transparency

            logger.debug(
                "Merge: dedup key=%s had %d matches, avg_dist=%.4f",
                key,
                len(group),
                avg_distance,
            )
            merged.append(merged_result)

    # Step 3: Global sort by distance (ascending — closer = more relevant)
    merged.sort(key=lambda r: r.get("distance", 0.0) or 0.0)

    logger.info(
        "Merge: %d total → %d unique results",
        len(results),
        len(merged),
    )

    # Step 4: Convert back to SearchResult objects (callers expect the same
    # type they passed in, not the internal dict representation).
    return [_dict_to_result(r) for r in merged]


# ============================================================
# Conversion helpers — SearchResult ↔ dict
# ============================================================


def _result_to_dict(result: Any) -> dict[str, Any]:
    """Convert a result object to a dict.

    Handles both dict-based and object-based (SearchResult-like) inputs.

    Args:
        result: A search result object or dictionary.

    Returns:
        Dictionary representation with keys: text, metadata, distance, collection.
    """
    if isinstance(result, dict):
        return result

    # Handle SearchResult-like dataclass/objects
    result_dict: dict[str, Any] = {}
    for attr in ("text", "metadata", "distance", "collection"):
        val = getattr(result, attr, None)
        if val is not None:
            result_dict[attr] = val

    return result_dict


def _dict_to_result(d: dict[str, Any]) -> Any:
    """Convert a dict back to the appropriate SearchResult type.

    Args:
        d: Dictionary representation of a search result.

    Returns:
        SearchResult object if distance/collection keys present, else the dict.
    """
    if "distance" in d or "collection" in d:
        try:
            from core.models import SearchResult
            metadata = d.get("metadata", {}) or {}
            sources_data = d.pop("_sources", None)

            # Convert _sources back to SearchResult objects if present
            sources = None
            if sources_data:
                sources = [_dict_to_result(s) for s in sources_data]

            return SearchResult(
                text=d.get("text", ""),
                metadata=metadata,
                distance=float(d.get("distance", 0.0)),
                collection=d.get("collection", ""),
                _sources=sources,
            )
        except ImportError:
            pass
        except TypeError as exc:
            # _sources constructor param may not exist — fall back without it
            logger.debug("SearchResult init failed (%s), falling back to dict", exc)

    return d


# Alias for external imports that expect the private name
_merge_results = merge_results

__all__ = ["merge_results", "_merge_results"]
