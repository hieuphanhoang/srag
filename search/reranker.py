"""SRAG Search Reranker — LLM-based reranking (DD-17, FD-75…FD-77).

Implements relevance-based reranking of search results using an LLM judge.
Each result is scored against the query, then sorted by descending score.

Falls back to original order when the LLM provider is unavailable or fails.
"""

from __future__ import annotations

import logging
from typing import Any

from llm.factory import create_llm_provider

logger = logging.getLogger(__name__)


class Reranker:
    """LLM-based reranker for search results.

    Uses a cross-encoder style prompt to score each result's relevance
    to the query, then returns results sorted by descending relevance score.

    Configuration:
        The LLM provider is determined from config.yaml under `llm.rerank`.
        Format: "provider/model" e.g. "anthropic/claude-sonnet-5" or "ollama/qwen3:4b".
        If not configured, falls back to original order.

    Thread safety:
        Not thread-safe by default. Each caller should create its own instance
        or use appropriate locking.
    """

    def __init__(self, llm_spec: str | None = None) -> None:
        """Initialize the reranker.

        Args:
            llm_spec: Optional LLM provider specification in "provider/model" format.
                     If None, attempts to read from config.yaml under `llm.rerank`.
        """
        self._llm_spec = llm_spec
        self._client: Any | None = None

    def _get_client(self) -> Any | None:
        """Get or create the LLM client.

        Returns:
            The LLM client, or None if unavailable.
        """
        if self._client is not None:
            return self._client

        spec = self._llm_spec
        if spec is None:
            # Read from config. Config is a flat dataclass (llm_rewrite,
            # llm_rerank, ...) - there is no nested cfg.llm.rerank.
            try:
                from config import load_config
                cfg = load_config()
                spec = getattr(cfg, 'llm_rerank', None)
            except Exception as exc:
                logger.debug("Failed to load config for reranker: %s", exc)

        if spec is not None:
            try:
                provider_name, model = spec.split("/", 1)
                self._client = create_llm_provider(provider_name, model=model)
                return self._client
            except ValueError as exc:
                logger.warning("Reranker LLM client creation failed: %s", exc)
            except Exception as exc:
                logger.debug("Unexpected error creating reranker client: %s", exc)

        return None

    def rerank(self, query: str, results: list[Any], top_k: int = 10) -> list[Any]:
        """Rerank search results by relevance to the query.

        Uses an LLM judge to score each result's relevance to the original query.
        Results are sorted by descending relevance score.

        Args:
            query: The original user query string.
            results: List of search result objects. Each must have a `text` attribute
                    (or be a dict with a 'text' key).
            top_k: Number of results to return after reranking. If the number of
                  input results is less than or equal to top_k, all are returned.

        Returns:
            Results sorted by descending relevance score. Always returns exactly
            min(len(results), top_k) results (or len(results) if fewer than top_k).

        Fallback behavior:
            If the LLM provider is unavailable or fails, returns results in their
            original order (no reranking applied).
        """
        if not query or not results:
            return list(results)

        # Normalize results to dicts for JSON serialization
        normalized = []
        for r in results:
            if isinstance(r, dict):
                normalized.append(r)
            else:
                # Convert SearchResult-like objects to dict representation
                normalized.append(self._result_to_dict(r))

        effective_k = min(top_k, len(normalized))

        # If fewer results than top_k, no need to rerank
        if len(normalized) <= effective_k:
            return [self._dict_to_result(d) for d in normalized]

        client = self._get_client()
        if client is None:
            logger.info("Reranker: LLM unavailable, returning original order")
            # Return top results by original distance (first items)
            return [self._dict_to_result(d) for d in normalized[:effective_k]]

        try:
            scores = self._score_results(client, query, normalized)
            if scores is None:
                logger.warning("Reranker: LLM scoring failed, returning original order")
                return [self._dict_to_result(d) for d in normalized[:effective_k]]

            # Pair results with their scores
            scored = list(zip(normalized, scores))

            # Sort by score descending (higher = more relevant)
            scored.sort(key=lambda x: x[1], reverse=True)

            # Take top_k
            top_results = [self._dict_to_result(d) for d, _ in scored[:effective_k]]

            logger.info(
                "Reranker: scored %d results, returned top %d",
                len(results),
                len(top_results),
            )
            return top_results

        except Exception as exc:
            logger.warning("Reranker failed (%s), returning original order", exc)
            return [self._dict_to_result(d) for d in normalized[:effective_k]]

    def _score_results(
        self, client: Any, query: str, results: list[dict]
    ) -> list[float] | None:
        """Score each result's relevance to the query using the LLM.

        Args:
            client: The LLM client.
            query: The original user query.
            results: List of result dicts with 'text' key.

        Returns:
            List of float scores (one per result), or None on failure.
        """
        # Process in batches to avoid token limits
        batch_size = 10
        all_scores: list[float] = []

        for i in range(0, len(results), batch_size):
            batch = results[i : i + batch_size]
            batch_scores = self._score_batch(client, query, batch)
            if batch_scores is None:
                return None
            all_scores.extend(batch_scores)

        return all_scores

    def _score_batch(
        self, client: Any, query: str, batch: list[dict]
    ) -> list[float] | None:
        """Score a single batch of results.

        Args:
            client: The LLM client.
            query: The original user query.
            batch: A batch of result dicts to score together.

        Returns:
            List of scores for this batch, or None on failure.
        """
        system_prompt = (
            "You are a relevance scorer for search results. "
            "Given a query and a set of text passages, rate each passage's "
            "relevance to the query on a scale of 0-10."
            "\n\n"
            "IMPORTANT: You MUST respond with valid JSON only.\n"
            "Do NOT include markdown, code blocks, explanations, or anything else.\n"
            'Your response must be parseable by Python\'s json.loads().\n'
            'The response must be a JSON object with the key "scores" containing an array of numbers.'
        )

        user_prompt_parts: list[str] = [f"Query: {query}\n\nPassages:\n"]
        for idx, passage in enumerate(batch):
            text = passage.get("text", "") or ""
            # Truncate very long passages to avoid token limits
            truncated = text[:500] if len(text) > 500 else text
            user_prompt_parts.append(f"Passage {idx + 1}: {truncated}\n")

        user_prompt = "\n---\n".join(user_prompt_parts) + (
            "\n\nReturn scores as JSON: "
            '{"scores": [score1, score2, ...]}'
        )

        try:
            generation_result = client.generate(
                prompt=f"{system_prompt}\n\n{user_prompt}",
                max_tokens=256,
            )
        except Exception as exc:
            logger.warning("Reranker LLM call failed: %s", exc)
            return None

        if generation_result is None or not generation_result.text:
            return None

        try:
            import json
            parsed = json.loads(generation_result.text)
            scores = parsed.get("scores")
            if not isinstance(scores, list) or len(scores) != len(batch):
                logger.warning(
                    "Reranker: invalid scores format (expected %d scores, got %s)",
                    len(batch),
                    type(scores).__name__,
                )
                return None

            # Validate all scores are numeric
            float_scores: list[float] = []
            for s in scores:
                if isinstance(s, (int, float)):
                    float_scores.append(float(s))
                else:
                    logger.warning("Reranker: non-numeric score %r", s)
                    return None

            return float_scores

        except (json.JSONDecodeError, AttributeError):
            logger.warning("Reranker: failed to parse LLM response as JSON")
            return None

    @staticmethod
    def _result_to_dict(result: Any) -> dict[str, Any]:
        """Convert a result object to a dict."""
        if isinstance(result, dict):
            return result

        # Handle SearchResult-like dataclass/objects
        result_dict: dict[str, Any] = {}
        for attr in ("text", "metadata", "distance", "collection"):
            val = getattr(result, attr, None)
            if val is not None:
                result_dict[attr] = val

        # Include any other public attributes
        for attr in dir(result):
            if not attr.startswith("_") and attr not in result_dict:
                val = getattr(result, attr)
                if not callable(val):
                    result_dict[attr] = val

        return result_dict

    @staticmethod
    def _dict_to_result(d: dict[str, Any]) -> Any:
        """Convert a dict back to the appropriate result type."""
        # If it looks like a SearchResult, preserve its structure
        if "distance" in d or "collection" in d:
            try:
                from models import SearchResult
                metadata = d.get("metadata", {})
                return SearchResult(
                    text=d.get("text", ""),
                    metadata=metadata,
                    distance=float(d.get("distance", 0.0)),
                    collection=d.get("collection", ""),
                )
            except ImportError:
                pass
        return d


# Module-level convenience function (uses default config)
def rerank(results: list[Any], query: str, top_k: int = 10) -> list[Any]:
    """Convenience function to rerank results.

    Args:
        results: List of search result objects or dicts.
        query: The original user query.
        top_k: Number of results to return.

    Returns:
        Results sorted by descending relevance score.
    """
    reranker = Reranker()
    return reranker.rerank(query, results, top_k)


__all__ = ["Reranker", "rerank"]