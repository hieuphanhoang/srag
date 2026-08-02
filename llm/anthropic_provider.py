"""SRAG Anthropic LLM Provider — Phase 2 implementation (DD-16).

Provides text generation via the Anthropic Claude API.
Supports all Claude models accessible through the Anthropic Python SDK.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any


logger = logging.getLogger(__name__)


@dataclass
class _DefaultConfig:
    """Defaults for Anthropic provider configuration."""
    DEFAULT_MODEL: str = "claude-sonnet-5"


class AnthropicProvider:
    """Anthropic Claude LLM provider using the official SDK.

    Parameters
    ----------
    model : str | None
        Model name to use (e.g., ``"claude-sonnet-5"``, ``"claude-opus-4"``).
        Defaults to ``"claude-sonnet-5"`` if not specified.
    api_key : str | None
        Anthropic API key. If ``None``, reads from the ``ANTHROPIC_API_KEY``
        environment variable. Required.

    Examples
    --------
    >>> provider = AnthropicProvider(api_key="sk-ant-...")
    >>> result = provider.generate("Explain RAG in one sentence.")
    >>> print(result.text[:100])
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        """Initialize the Anthropic provider."""
        try:
            import anthropic  # noqa: F401 — validates SDK is available
        except ImportError as exc:
            raise ImportError(
                "The 'anthropic' package is required for AnthropicProvider. "
                "Install it with: pip install anthropic"
            ) from exc

        self._model: str = model or _DefaultConfig.DEFAULT_MODEL

        # api_key can be passed directly or read from env by the SDK internally.
        self._client = anthropic.Anthropic(api_key=api_key)
        logger.info("Anthropic provider initialized: model=%s", self._model)

    def generate(self, prompt: str, max_tokens: int = 512, **kwargs: Any):
        """Generate text from a single prompt using Claude.

        Parameters
        ----------
        prompt : str
            The input prompt to send to Claude.
        max_tokens : int
            Maximum number of tokens to generate. Defaults to 512.
        **kwargs
            Additional parameters passed to the Anthropic API (e.g., ``temperature``,
            ``stop_sequences``, ``system``).

        Returns
        -------
        GenerationResult
            The generation result with text, model info, and usage stats.

        Raises
        ------
        ValueError
            If *prompt* is empty or *api_key* is missing.
        RuntimeError
            If the Anthropic API call fails.
        """
        from .base import GenerationResult

        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        # Validate API key is available.
        if not self._client.api_key:
            raise ValueError(
                "Anthropic API key is required. Pass api_key=... or set "
                "the ANTHROPIC_API_KEY environment variable."
            )

        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )

            # Extract result from Anthropic response.
            text = ""
            if hasattr(response, "content") and response.content:
                blocks = response.content
                text = "\n".join(
                    block.text for block in blocks if hasattr(block, "text")
                )

            usage = None
            if hasattr(response, "usage"):
                u = response.usage
                usage = {
                    "input_tokens": u.input_tokens if hasattr(u, "input_tokens") else 0,
                    "output_tokens": u.output_tokens if hasattr(u, "output_tokens") else 0,
                }

            return GenerationResult(
                text=text,
                model=self._model,
                usage=usage,
            )
        except Exception as exc:
            logger.error("Anthropic generation failed: %s", exc)
            raise RuntimeError(f"Anthropic generate error: {exc}") from exc

    def generate_batch(
        self,
        prompts: list[str],
        max_tokens: int = 512,
        **kwargs: Any,
    ) -> list:
        """Generate text for multiple prompts sequentially.

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
                logger.error("Anthropic batch generate failed for prompt %d: %s", i, exc)
                results.append(GenerationResult(text="", model=self._model))

        return results

    @property
    def model_name(self) -> str:
        return self._model

    def __repr__(self) -> str:  # noqa: D105
        return f"AnthropicProvider(model={self._model!r})"


# Alias for test compatibility
AnthropicLLMProvider = AnthropicProvider
