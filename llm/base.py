"""SRAG LLM Base — Phase 2 implementation (DD-13, DD-14).

LLM provider protocol base class and abstract interface for all LLM providers.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result of a text generation request.

    Attributes
    ----------
    text : str
        The generated text response.
    model : str | None
        The model name that produced this result, if available.
    usage : dict[str, int] | None
        Token usage statistics (e.g., ``{"prompt_tokens": 10, "completion_tokens": 50}``).
    """

    text: str
    model: str | None = None
    usage: dict[str, int] | None = None


class LLMProvider(ABC):
    """Abstract base class for all LLM providers.

    Each concrete provider (Ollama, Anthropic, etc.) implements the
    :meth:`generate` method using its own API protocol.
    """

    def __init__(self, model: str) -> None:
        """Initialize the LLM provider.

        Parameters
        ----------
        model : str
            The model name/identifier to use (e.g., ``"qwen3:4b"``,
            ``"claude-sonnet-5"``).
        """
        self.model = model

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 512, **kwargs: Any) -> GenerationResult:
        """Generate text from a prompt.

        Parameters
        ----------
        prompt : str
            The input prompt to send to the model.
        max_tokens : int
            Maximum number of tokens to generate. Defaults to 512.
        **kwargs
            Provider-specific parameters (e.g., ``temperature``, ``stop``).

        Returns
        -------
        GenerationResult
            The generation result containing text, model info, and usage stats.

        Raises
        ------
        ValueError
            If *prompt* is empty or invalid.
        RuntimeError
            If the provider API call fails (network error, auth failure, etc.).
        """
        ...

    @abstractmethod
    def generate_batch(
        self,
        prompts: list[str],
        max_tokens: int = 512,
        **kwargs: Any,
    ) -> list[GenerationResult]:
        """Generate text for multiple prompts in a batch.

        Parameters
        ----------
        prompts : list[str]
            List of input prompts.
        max_tokens : int
            Maximum number of tokens per generation. Defaults to 512.
        **kwargs
            Provider-specific parameters.

        Returns
        -------
        list[GenerationResult]
            One result per prompt, in the same order. Failed generations
            return ``GenerationResult(text="")`` with an error logged.
        """
        ...

    def __repr__(self) -> str:  # noqa: D105
        return f"{self.__class__.__name__}(model={self.model!r})"


import abc
from typing import Any, Dict


class LLMClient(abc.ABC):
    """Minimal async-style LLM client interface used by enrichment & search modules.

    The ``call`` method accepts structured message arguments and returns a
    dict with at least a ``"text"`` key containing the model's response.
    """

    @abc.abstractmethod
    def call(
        self,
        system: str,
        user_message: str,
        json_mode: bool = False,
        timeout_ms: int | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Send a request to the LLM and return the parsed response.

        Parameters
        ----------
        system : str
            System prompt.
        user_message : str
            User message content.
        json_mode : bool
            If True, request JSON-only output. Defaults to False.
        timeout_ms : int | None
            Timeout in milliseconds. None means no timeout.
        **kwargs
            Additional provider-specific parameters.

        Returns
        -------
        Dict[str, Any]
            Must contain ``"text"`` key with the response text.
        """
        ...


# Re-export for convenience.
__all__ = ["LLMProvider", "GenerationResult", "LLMClient"]
