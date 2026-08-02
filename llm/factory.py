"""SRAG LLM Factory — Phase 2 implementation (DD-07).

Factory for creating LLM provider instances from configuration.
Supports Ollama and Anthropic providers via config.yaml settings.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import LLMProvider

logger = logging.getLogger(__name__)


def create_llm_provider(
    provider_name: str,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> LLMProvider:
    """Create an LLM provider instance.

    Parameters
    ----------
    provider_name : str
        Provider identifier (``"ollama"`` or ``"anthropic"``, case-insensitive).
    model : str | None
        Model name to use. If ``None``, falls back to config.yaml defaults.
    base_url : str | None
        Base URL for the provider API. For Ollama, typically
        ``"http://localhost:11434/v1"``. If ``None``, uses default or config.
    api_key : str | None
        API key for authentication. Required for Anthropic. Not used by local
        Ollama instances but may be needed for remote Ollama deployments.

    Returns
    -------
    LLMProvider
        A configured provider instance.

    Raises
    ------
    ValueError
        If *provider_name* is unknown or required parameters are missing.
    RuntimeError
        If the provider cannot be initialized (e.g., connection failure).
    """
    name = provider_name.lower().strip()

    if name == "ollama" or name == "oll":
        from .ollama_provider import OllamaProvider

        return OllamaProvider(
            model=model,
            base_url=base_url,
            api_key=api_key,
        )

    if name == "anthropic" or name == "ant":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider(
            model=model,
            api_key=api_key,
        )

    raise ValueError(
        f"Unknown LLM provider '{provider_name}'. Supported: 'ollama', 'anthropic'"
    )


def create_client(spec: str | None = None) -> LLMProvider:
    """Create an LLM provider from a "provider/model" spec string.

    This is a convenience wrapper around :func:`create_llm_provider` for use by
    search modules (rewriter, reranker).

    Parameters
    ----------
    spec : str | None
        LLM specification in ``"provider/model"`` format, e.g.
        ``"ollama/qwen3:4b"`` or ``"anthropic/claude-sonnet-5"``.
        If ``None``, falls back to config.yaml defaults.

    Returns
    -------
    LLMProvider
        A configured provider instance.

    Raises
    ------
    ValueError
        If *spec* is invalid or required parameters are missing.
    """
    if spec is None:
        # Fall back to config.yaml
        from .ollama_provider import OllamaProvider
        return OllamaProvider()

    parts = spec.split("/", 1)
    if len(parts) != 2:
        raise ValueError(
            f"Invalid LLM spec '{spec}'. Expected format: 'provider/model'"
        )
    provider_name, model = parts

    # Load config for optional overrides. Config is a flat dataclass (no
    # nested `.llm` attribute), so pull the field that's relevant to the
    # specific provider being constructed.
    base_url = None
    api_key = None
    try:
        from core.config import load_config
        cfg = load_config()
        name = provider_name.lower().strip()
        if name in ("ollama", "oll"):
            base_url = cfg.ollama_url
        elif name in ("anthropic", "ant"):
            api_key = cfg.anthropic_api_key or None
    except Exception:
        pass

    return create_llm_provider(
        provider_name=provider_name,
        model=model,
        base_url=base_url,
        api_key=api_key,
    )


# Alias for compatibility with existing code that imports get_provider
get_provider = create_llm_provider

__all__ = ["create_llm_provider", "create_client", "get_provider"]
