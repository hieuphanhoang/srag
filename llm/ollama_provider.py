"""SRAG Ollama LLM Provider — Phase 2 implementation (DD-15).

Provides text generation via the local `ollama` Python package (same
dependency already used by `embedding.py`). Supports both local and remote
Ollama instances.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class _DefaultConfig:
    """Defaults for Ollama provider configuration."""
    DEFAULT_BASE_URL: str = "http://localhost:11434"
    DEFAULT_MODEL: str = "qwen3:4b"


class OllamaProvider:
    """Ollama LLM provider using the `ollama` package's chat API.

    Parameters
    ----------
    model : str | None
        Model name to use (e.g., ``"qwen3:4b"``, ``"llama3.1:8b"``).
        Defaults to ``"qwen3:4b"`` if not specified.
    base_url : str | None
        Ollama API base URL. Defaults to ``"http://localhost:11434"``.
    api_key : str | None
        Unused for local Ollama instances; accepted for interface parity
        with other providers (remote/managed Ollama deployments that sit
        behind auth can pass a bearer token here).

    Examples
    --------
    >>> provider = OllamaProvider(model="qwen3:4b")
    >>> result = provider.generate("What is RAG?")
    >>> print(result.text[:100])
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        """Initialize the Ollama provider."""
        import ollama as _ollama

        self._model: str = model or _DefaultConfig.DEFAULT_MODEL
        url = base_url or _DefaultConfig.DEFAULT_BASE_URL

        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        self._client = _ollama.Client(host=url, headers=headers)
        logger.info("Ollama provider initialized: model=%s, url=%s", self._model, url)

    def generate(self, prompt: str, max_tokens: int = 512, **kwargs: Any):
        """Generate text from a single prompt.

        Parameters
        ----------
        prompt : str
            The input prompt to send to the model.
        max_tokens : int
            Maximum number of tokens to generate. Defaults to 512.
        **kwargs
            Additional parameters merged into Ollama's ``options`` dict
            (e.g., ``temperature``, ``stop``).

        Returns
        -------
        GenerationResult
            The generation result with text, model info, and usage stats.

        Raises
        ------
        ValueError
            If *prompt* is empty.
        RuntimeError
            If the Ollama API call fails.
        """
        from .base import GenerationResult

        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        # Thinking-capable models (e.g. the qwen3 family) return their
        # chain-of-thought in a separate `message.thinking` field and can
        # exhaust the entire max_tokens budget on it, leaving
        # `message.content` (what callers actually read) empty. Every
        # current caller (query rewriting, reranking, enrichment) wants a
        # direct answer, not exposed reasoning, so default thinking off;
        # still overridable via an explicit `think=` kwarg.
        think = kwargs.pop("think", False)

        try:
            response = self._client.chat(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                think=think,
                options={"num_predict": max_tokens, **kwargs},
            )

            message = response.get("message", {}) if isinstance(response, dict) else response.message
            text = message.get("content", "") if isinstance(message, dict) else (message.content or "")

            prompt_tokens = response.get("prompt_eval_count") if isinstance(response, dict) else getattr(response, "prompt_eval_count", None)
            completion_tokens = response.get("eval_count") if isinstance(response, dict) else getattr(response, "eval_count", None)
            usage = None
            if prompt_tokens is not None or completion_tokens is not None:
                usage = {
                    "prompt_tokens": prompt_tokens or 0,
                    "completion_tokens": completion_tokens or 0,
                }

            return GenerationResult(
                text=text,
                model=self._model,
                usage=usage,
            )
        except Exception as exc:
            logger.error("Ollama generation failed: %s", exc)
            raise RuntimeError(f"Ollama generate error: {exc}") from exc

    def generate_batch(
        self,
        prompts: list[str],
        max_tokens: int = 512,
        **kwargs: Any,
    ) -> list:
        """Generate text for multiple prompts.

        Parameters
        ----------
        prompts : list[str]
            List of input prompts.
        max_tokens : int
            Maximum tokens per generation. Defaults to 512.
        **kwargs
            Additional parameters passed to the API.

        Returns
        -------
        list[GenerationResult]
            One result per prompt, in order. Failed generations return an empty
            ``GenerationResult(text="")`` with error logged.
        """
        from .base import GenerationResult

        results: list[GenerationResult] = []
        for i, prompt in enumerate(prompts):
            try:
                result = self.generate(prompt, max_tokens=max_tokens, **kwargs)
                results.append(result)
            except Exception as exc:
                logger.error("Ollama batch generate failed for prompt %d: %s", i, exc)
                results.append(GenerationResult(text="", model=self._model))

        return results

    @property
    def model_name(self) -> str:
        return self._model

    def __repr__(self) -> str:  # noqa: D105
        return f"OllamaProvider(model={self._model!r})"


# Alias for test compatibility
OllamaLLMProvider = OllamaProvider
