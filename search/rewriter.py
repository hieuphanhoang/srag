"""SRAG Query Rewriter — LLM-based query expansion for search optimization.

Implements cross-lingual (EN↔VI) query rewriting using an LLM to expand
and clarify the user's query before vector search. Addresses DD-14, FD-32...FD-37.

Key design decisions:
- Uses the same LLM client factory as other modules for consistency.
- Returns multiple rewrite candidates from a single LLM call via JSON output.
- Falls back to returning the original query if rewriting fails.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from llm.factory import create_llm_provider

logger = logging.getLogger(__name__)


class QueryRewriter:
    """LLM-based query rewriter for search optimization.

    Expands and clarifies user queries using an LLM to generate rewrite
    candidates. Supports cross-lingual expansion (EN/VI) via system prompt.

    Configuration:
        The LLM provider is determined from config.yaml under `llm.search`.
        Format: "provider/model" e.g. "anthropic/claude-sonnet-5".
        If not configured, falls back to returning the original query as-is.

    Thread safety:
        Not thread-safe by default. Each caller should create its own instance
        or use appropriate locking.
    """

    def __init__(self, llm_spec: str | None = None) -> None:
        """Initialize the query rewriter.

        Args:
            llm_spec: Optional LLM provider specification in "provider/model" format.
                     If None, attempts to read from config.yaml under `llm.search`.
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
            # Read from config.yaml. Config is a flat dataclass (llm_rewrite,
            # llm_rerank, ...) - there is no nested cfg.llm.search.
            try:
                from config import load_config
                cfg = load_config()
                spec = getattr(cfg, 'llm_rewrite', None)
            except Exception as exc:
                logger.debug("Failed to load config for rewriter: %s", exc)

        if spec is not None:
            try:
                provider_name, model = spec.split("/", 1)
                self._client = create_llm_provider(provider_name, model=model)
                return self._client
            except ValueError as exc:
                logger.warning("QueryRewriter LLM client creation failed: %s", exc)
            except Exception as exc:
                logger.debug("Unexpected error creating rewriter client: %s", exc)

        return None

    def rewrite(self, query: str, max_candidates: int = 3) -> list[str]:
        """Rewrite a query for search optimization.

        Uses an LLM to generate expanded/clarified versions of the user's query.
        Supports cross-lingual expansion (English / Vietnamese).

        Args:
            query: The original user query string.
            max_candidates: Maximum number of rewrite candidates to return.
                          Default is 3. Includes the original query.

        Returns:
            List of rewritten queries including the original as the first element.
            Always returns at least [query] even if rewriting fails.
            Example: ["original query", "expanded variant 1", "cross-lingual variant"]

        Fallback behavior:
            If the LLM provider is unavailable or fails, returns [query].
        """
        if not query or not query.strip():
            return []

        # Normalize: strip whitespace
        original_query = query.strip()

        # Always include the original query as first candidate
        candidates: list[str] = [original_query]

        client = self._get_client()
        if client is None:
            logger.info("QueryRewriter: LLM unavailable, returning original query only")
            return candidates

        try:
            rewritten = self._rewrite_with_llm(client, original_query, max_candidates)
            if rewritten:
                # Filter out duplicates (case-insensitive, stripped)
                seen = {q.lower().strip() for q in candidates}
                for r in rewritten:
                    normalized = r.strip()
                    if normalized and normalized.lower() not in seen:
                        candidates.append(normalized)
                        seen.add(normalized.lower())

            logger.info(
                "QueryRewriter: generated %d candidate(s) for query: %r",
                len(candidates) - 1,
                original_query,
            )

        except Exception as exc:
            logger.warning("QueryRewriter failed (%s), returning original query", exc)

        return candidates

    def _rewrite_with_llm(
        self,
        client: Any,
        query: str,
        max_candidates: int,
    ) -> list[str]:
        """Generate rewrite candidates using the LLM.

        Args:
            client: The LLM client (already validated as non-None).
            query: The original user query.
            max_candidates: Maximum number of rewritten variants to generate
                          (excluding the original query).

        Returns:
            List of rewritten queries from the LLM, or empty list on failure.
        """
        system_prompt = (
            "You are a query expansion specialist for information retrieval. "
            "Your task is to rewrite user queries to improve search recall."
            "\n\n"
            "Capabilities:"
            " 1. Cross-lingual expansion: translate between English and Vietnamese."
            "   - VI->EN: Provide an English translation of the query."
            "   - EN->VI: Provide a Vietnamese translation AND expanded terms."
            "2. Synonym/paraphrase: provide alternative phrasings in the same language."
            "3. Term expansion: add related technical terms or acronyms."
            "4. Generalization: broaden narrow queries for wider coverage."
            "\n\n"
            "IMPORTANT: You MUST respond with valid JSON only.\n"
            "Do NOT include markdown, code blocks, explanations, or anything else.\n"
            'Your response must be parseable by Python\'s json.loads().\n'
            'The response must be a JSON object with the key "rewrites" containing an array of strings.'
        )

        # Determine source language hint for the LLM
        if _is_vietnamese(query):
            lang_hint = (
                "The user's query is in Vietnamese. Please provide:\n"
                " 1. English translation\n"
                "2. Vietnamese synonyms/expansions\n"
                "3. Related technical terms (in both EN and VI if applicable)"
            )
        else:
            lang_hint = (
                "The user's query is in English. Please provide:\n"
                " 1. Vietnamese translation\n"
                "2. English synonyms/expansions\n"
                "3. Related technical terms (in both EN and VI if applicable)"
            )

        user_prompt = (
            f"{lang_hint}"
            "\n\nQuery: {query}\n\n"
            "Return up to {max_candidates} rewrite candidates as JSON:\n"
            '{{"rewrites": ["rewrite1", "rewrite2", ...]}}'
        ).format(query=query, max_candidates=max_candidates)

        try:
            generation_result = client.generate(
                prompt=f"{system_prompt}\n\n{user_prompt}",
                max_tokens=512,
            )
        except Exception as exc:
            logger.warning("QueryRewriter LLM call failed: %s", exc)
            return []

        if generation_result is None or not generation_result.text:
            return []

        try:
            import json
            parsed = json.loads(generation_result.text)
            rewrites = parsed.get("rewrites")

            if not isinstance(rewrites, list):
                logger.warning("QueryRewriter: invalid rewrites format (expected list)")
                return []

            # Validate all items are strings and convert to float-compatible values
            result: list[str] = []
            for item in rewrites[:max_candidates]:  # Limit to max_candidates
                if isinstance(item, str) and item.strip():
                    result.append(item.strip())

            return result

        except (json.JSONDecodeError, AttributeError):
            logger.warning("QueryRewriter: failed to parse LLM response as JSON")
            return []


# ============================================================
# Language detection helpers
# ============================================================

# Vietnamese-specific precomposed characters (NFC -- how real Vietnamese
# text is actually encoded, not the NFD combining-mark sequences the
# previous heuristic assumed, which never match real-world input). Covers
# accented Latin-1 vowels, the Vietnamese-only base letters (Ă/ă
# Đ/đ Ĩ/ĩ Ũ/ũ Ơ/ơ Ư/ư), and
# the Latin Extended Additional block (U+1EA0-U+1EF9), which carries the
# tone-marked vowels essentially unique to Vietnamese orthography.
_VI_CHAR_RE = re.compile(
    "[À-ÃÈ-ÊÌ-ÍÐÒ-ÕÙ-ÚÝ"
    "à-ãè-êì-íðò-õù-úý"
    "ĂăĐđĨĩŨũƠơƯư"
    "Ạ-ỹ]"
)

# Common Vietnamese function words (typed without diacritics), matched as
# whole words only -- avoids false positives from short strings that merely
# appear as a *substring* of unrelated English words (the previous
# heuristic used bare `in` substring checks against single letters like
# "l"/"d"/"e" and even the English word "the", so ordinary English text
# routinely misdetected as Vietnamese).
_VI_WORDS = {
    "la", "luon", "khong", "co", "day", "do", "vi", "cua", "va",
    "truong", "hoc", "sinh", "nha", "rung", "tu",
}

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def _is_vietnamese(text: str) -> bool:
    """Detect if text is primarily Vietnamese.

    Uses character-level heuristics (Vietnamese-specific precomposed
    diacritics) and whole-word frequency analysis to determine the language.

    Args:
        text: The text to analyze.

    Returns:
        True if the text appears to be Vietnamese, False otherwise.
    """
    if not text or not text.strip():
        return False

    # Strong signal: presence of a Vietnamese-specific precomposed character.
    if _VI_CHAR_RE.search(text):
        return True

    # Weak signal: whole-word matches against common Vietnamese function words.
    words = set(_WORD_RE.findall(text.lower()))
    vi_word_matches = sum(1 for w in _VI_WORDS if w in words)

    if vi_word_matches >= 3:
        return True

    return False


# Module-level convenience function (uses default config)
def rewrite(query: str, max_candidates: int = 3) -> list[str]:
    """Convenience function to rewrite a query.

    Args:
        query: The original user query string.
        max_candidates: Maximum number of rewrite candidates to return.
                       Default is 3.

    Returns:
        List of rewritten queries including the original.
    """
    rewriter = QueryRewriter()
    return rewriter.rewrite(query, max_candidates)


__all__ = ["QueryRewriter", "rewrite"]
