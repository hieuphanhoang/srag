"""
Enrichment module for semantic RAG.

Provides text enrichment via LLM with retry, backoff, and cancellation support.
Implements DD-10 (per-chunk enrichment with exponential backoff) and DD-11
(batch enrichment with ThreadPoolExecutor concurrency).
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from typing import List, Optional, Union

from .base import LLMClient
from .factory import create_client
from ingest.retry import retry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enrichment schema constants (matches FD-55 / DD-10 specification)
# ---------------------------------------------------------------------------

ENRICHMENT_SYSTEM_PROMPT = """\
You are a text analysis assistant. Given a block of text, analyze it and produce structured metadata.

Return a JSON object with these EXACT fields:
- `headline`: A short descriptive title (max 100 characters).
- `summary`: A 1-2 sentence summary capturing the main idea.
- `keywords`: An array of 3-8 relevant keywords or key phrases (strings).
- `entities`: An array of named entities mentioned (persons, organizations, locations, dates) (strings).
- `sentiment`: One of "positive", "negative", "neutral".
- `topics`: An array of 2-5 topic categories relevant to the text.

CRITICAL: Return ONLY valid JSON with no markdown formatting, no code fences, no explanation text. The output must be parseable by json.loads().
"""


# ---------------------------------------------------------------------------
# Data classes (matches FD-55 / DD-10 EnrichedChunk schema)
# ---------------------------------------------------------------------------

@dataclass
class EnrichedMetadata:
    """Enrichment result for a single chunk of text."""
    headline: str
    summary: str
    keywords: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    sentiment: str = "neutral"
    topics: List[str] = field(default_factory=list)

    def as_text(self) -> str:
        """Returns headline + summary + original_text for embedding.

        Note: `original_text` is set by the caller after enrichment.
        """
        parts = [self.headline, self.summary]
        if hasattr(self, '_original_text') and self._original_text:
            parts.append('\n' + self._original_text)
        return '\n\n'.join(parts)


class EnrichmentCancelled(Exception):
    """Raised when enrichment is cancelled."""
    pass


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _is_valid_enrichment(data: dict) -> bool:
    """Validate that the LLM response matches the expected enrichment schema."""
    required_fields = {'headline', 'summary'}
    if not isinstance(data, dict):
        return False
    if not required_fields.issubset(data.keys()):
        return False
    if not isinstance(data.get('headline'), str) or not data['headline']:
        return False
    if not isinstance(data.get('summary'), str) or not data['summary']:
        return False
    # Optional fields: validate types if present
    for field_name, expected_type in [
        ('keywords', list),
        ('entities', list),
        ('sentiment', str),
        ('topics', list),
    ]:
        if field_name in data:
            val = data[field_name]
            if not isinstance(val, expected_type):
                return False
    # Validate sentiment value
    if 'sentiment' in data and data['sentiment'] not in ('positive', 'negative', 'neutral'):
        return False
    # Validate headline length
    if len(data.get('headline', '')) > 100:
        return False
    return True


# ---------------------------------------------------------------------------
# EnrichmentClient (implements DD-10 enrich + DD-11 enrich_batch)
# ---------------------------------------------------------------------------

class EnrichmentClient:
    """Text enrichment via LLM with retry, backoff, and cancellation."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        if llm_client is not None:
            self._client = llm_client
            return

        # create_client(None) falls back to OllamaProvider's hardcoded
        # default model (qwen3:4b), silently ignoring config.yaml's
        # `llm.enrichment` spec — resolve it explicitly, same pattern
        # QueryRewriter/Reranker use for `llm.search`/`llm.rerank`.
        spec = None
        try:
            from core.config import load_config
            spec = load_config().llm_enrichment
        except Exception as exc:
            logger.debug("Failed to load config for enrichment client: %s", exc)

        try:
            self._client = create_client(spec)
        except Exception as exc:
            logger.warning("Enrichment LLM client creation failed (%s); falling back to default.", exc)
            self._client = create_client(None)

    @property
    def client(self) -> LLMClient:
        return self._client

    # ------------------------------------------------------------------
    # DD-10: enrich(text, cancel_flag=None) → EnrichedMetadata | None
    # ------------------------------------------------------------------

    @retry(max_retries=3, base_delay=1.0, retry_on=(TimeoutError, ConnectionError, Exception))
    def enrich(self, text: str, cancel_flag: Optional[list] = None) -> Optional[EnrichedMetadata]:
        """Enrich a single chunk of text with LLM-generated metadata.

        Implements DD-10 enrichment retry with exponential backoff.

        Args:
            text: The raw text to enrich.
            cancel_flag: If set and truthy, signals cancellation.

        Returns:
            EnrichedMetadata on success, None if enrichment fails after retries
            or if the LLM response is invalid.

        Raises:
            EnrichmentCancelled: If cancel_flag is set.
        """
        # Check cancellation before proceeding
        if cancel_flag and cancel_flag():
            raise EnrichmentCancelled("Enrichment cancelled")

        # Build prompt for the LLM
        user_message = f"Enrich this text:\n\n{text}"

        try:
            # LLMProvider (Ollama/Anthropic) exposes generate(prompt, max_tokens),
            # not the LLMClient.call(system, user_message, json_mode) interface —
            # combine system + user into a single prompt like search/rewriter.py
            # and search/reranker.py already do against the same providers.
            response = self._client.generate(
                prompt=f"{ENRICHMENT_SYSTEM_PROMPT}\n\n{user_message}",
                max_tokens=512,
            )

            raw_text = response.text or ''

            # Parse JSON response
            enriched_data = json.loads(raw_text)

        except (json.JSONDecodeError, ValueError):
            # Validation errors: malformed LLM output → give up immediately
            logger.warning("Enrichment JSON parse failed for text starting with: %s...", text[:100])
            return None

        except Exception as e:
            # Transient errors handled by @retry decorator
            raise

        # Validate schema
        if not _is_valid_enrichment(enriched_data):
            logger.warning("Enrichment validation failed for text starting with: %s...", text[:100])
            return None  # give up, don't retry (schema mismatch)

        # Build result object; store original_text as private attribute
        enriched = EnrichedMetadata(**enriched_data)
        enriched._original_text = text  # type: ignore[attr-defined]
        return enriched

    # ------------------------------------------------------------------
    # DD-11: enrich_batch(texts, cancel_flag=None) → list[EnrichedChunk | None]
    # ------------------------------------------------------------------

    def enrich_batch(
        self,
        texts: List[str],
        cancel_flag: Optional[callable] = None,
    ) -> List[Optional[EnrichedMetadata]]:
        """Enrich a batch of texts with LLM-generated metadata.

        Implements DD-11 batch enrichment with ThreadPoolExecutor concurrency.

        Args:
            texts: List of raw text chunks to enrich.
            cancel_flag: If set and truthy, signals cancellation between futures.

        Returns:
            List of EnrichedMetadata (or None on failure) for each input text.
        """
        if not texts:
            return []

        results: List[Optional[EnrichedMetadata]] = [None] * len(texts)

        executor = ThreadPoolExecutor(max_workers=3)  # DD-11: 3 concurrent workers
        futures = []

        try:
            # Submit all tasks (eager submission)
            for idx, text in enumerate(texts):
                future = executor.submit(self.enrich, text, cancel_flag)
                futures.append((idx, future))

            # Collect results with cancellation checks
            for idx, future in futures:
                # Check cancellation before collecting each result
                if cancel_flag and cancel_flag():
                    raise EnrichmentCancelled("Enrichment cancelled during batch collection")

                try:
                    enriched = future.result(timeout=35)  # DD-11: 35s timeout per future
                    results[idx] = enriched
                except FuturesTimeoutError:
                    logger.error("Enrichment for text %d timed out", idx)
                    results[idx] = None
                except EnrichmentCancelled:
                    # Propagate cancellation upward
                    executor.shutdown(wait=False)
                    raise
                except Exception as e:
                    logger.error("Enrichment for text %d failed: %s", idx, e)
                    results[idx] = None

        finally:
            # Always shutdown the executor
            executor.shutdown(wait=True)

        return results