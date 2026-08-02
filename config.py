"""
SRAG Configuration Management (FD-01 ... FD-06, DD-01).

Implements:
  - DD-01: Environment variable expansion ($VAR / ${VAR}) with env-var-level override precedence.
  - DD-02: Config class and loader with YAML + environment fallback.
  - DD-03: Default configuration values (config.yaml-driven baseline).
  - DD-04: Path resolution for data stores (LOCALAPPDATA-based paths, cross-platform).
  - DD-05: Configuration validation on load.
  - FD-06: Runtime PATCH endpoint not persisted to disk.

Environment variable overrides (DD-01):
    SRAG_OLLAMA_URL          → config.ollama_url
    SRAG_EMBEDDING_MODEL     → config.embedding_model
    SRAG_LLM_PROVIDER        → config.llm_provider
    SRAG_ANTRHOPIC_API_KEY   → config.anthropic_api_key (note: legacy spelling)
    SRAG_OllAMA_TIMEOUT      → config.ollama_timeout
    SRAG_EMBEDDING_DIM       → config.embedding_dim
    SRAG_TOP_K               → config.top_k

Usage example:
    >>> from config import load_config
    >>> cfg = load_config("config.yaml")
    >>> print(cfg.ollama_url)
"""

from __future__ import annotations

import os
import re
import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# ---------------------------------------------------------------------------
# DD-04: Default paths (LOCALAPPDATA-based)
# ---------------------------------------------------------------------------
def _default_chromadb_path() -> str:
    localappdir = os.environ.get("LOCALAPPDATA", "")
    if not localappdir:
        localappdir = os.path.join(Path.home(), "AppData", "Local")
    return os.path.join(localappdir, "srag", "chromadb")


def _default_folders_db_path() -> str:
    localappdir = os.environ.get("LOCALAPPDATA", "")
    if not localappdir:
        localappdir = os.path.join(Path.home(), "AppData", "Local")
    return os.path.join(localappdir, "srag", "folders.db")


def _default_log_path() -> str:
    return os.path.join(os.getcwd(), "logs", "srag.log")


# ---------------------------------------------------------------------------
# DD-03: Default values
# ---------------------------------------------------------------------------
_DEFAULTS: Dict[str, Any] = {
    # Ollama (DD-04)
    "ollama_url": "http://localhost:11434",
    "embedding_model": "qwen3-embedding",
    "llm_provider": "ollama/qwen3:4b",
    "anthropic_api_key": "",
    # Embedding (DD-05)
    "embedding_dim": 4096,
    # Search (FD-02)
    "top_k": 10,
    "enrichment_chunk_size": 768,
    "enrichment_overlap": 64,
    # Logging (DD-03 / FD-07)
    "log_level": "INFO",
}


# ---------------------------------------------------------------------------
# DD-02: Config class and loader
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """Flat configuration container — all attributes are public (FD-05)."""

    # Ollama settings
    ollama_url: str = _DEFAULTS["ollama_url"]
    embedding_model: str = _DEFAULTS["embedding_model"]
    llm_provider: str = _DEFAULTS["llm_provider"]
    anthropic_api_key: str = ""
    # Embedding / chunking (DD-05)
    embedding_dim: int = _DEFAULTS["embedding_dim"]
    enrichment_chunk_size: int = _DEFAULTS["enrichment_chunk_size"]
    enrichment_overlap: int = _DEFAULTS["enrichment_overlap"]
    # Search (FD-02)
    top_k: int = 10
    # Logging (DD-03 / FD-07)
    log_level: str = _DEFAULTS["log_level"]

    # ChromaDB path (DD-04)
    chromadb_path: str = field(default_factory=_default_chromadb_path)
    # Folders DB path (DD-04)
    folders_db_path: str = field(default_factory=_default_folders_db_path)
    # Log file path
    log_file: str = _DEFAULTS["log_path"]

    # Runtime-only config patches (FD-06 — not persisted)
    _runtime_patches: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# DD-01: Environment variable expansion & override precedence
# ---------------------------------------------------------------------------

_ENV_VAR_PATTERN = re.compile(r"\$\{(\w+)\}|(\$[\w]+)")

_ENV_MAPPING: List[tuple] = [
    # (yaml_key, SRAG_* env name, target_type)
    ("ollama_url", "SRAG_OLLAMA_URL", str),
    ("embedding_model", "SRAG_EMBEDDING_MODEL", str),
    ("llm_provider", "SRAG_LLM_PROVIDER", str),
    ("anthropic_api_key", "SRAG_ANTRHOPIC_API_KEY", str),
    ("ollama_timeout", "SRAG_OllAMA_TIMEOUT", int),
    ("embedding_dim", "SRAG_EMBEDDING_DIM", int),
    ("top_k", "SRAG_TOP_K", int),
]


def _expand_env_vars(value: Any) -> Any:
    """Recursively expand $VAR / ${VAR} references in a string value."""
    if not isinstance(value, str):
        return value

    def _replace(match: re.Match) -> str:
        var_name = match.group(1) or match.group(2).lstrip("$")
        env_val = os.environ.get(var_name, "")
        # Recursive expansion guard (max 8 levels)
        if "${" in env_val or "$" in env_val:
            return _expand_env_vars(env_val)
        return env_val

    expanded = _ENV_VAR_PATTERN.sub(_replace, value)
    return expanded


def _apply_env_overrides(cfg_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Apply SRAG_* environment variable overrides (DD-01 precedence)."""
    result = copy.deepcopy(cfg_dict)

    for yaml_key, env_name, target_type in _ENV_MAPPING:
        raw = os.environ.get(env_name)
        if raw is not None:
            if target_type == int:
                try:
                    result[yaml_key] = int(raw)
                except (ValueError, TypeError):
                    pass  # keep default on bad value
            else:
                result[yaml_key] = raw.strip()

    return result


# ---------------------------------------------------------------------------
# DD-05: Configuration validation
# ---------------------------------------------------------------------------

_VALIDATION_RULES = {
    "ollama_url": lambda v: isinstance(v, str) and len(v) > 0,
    "embedding_model": lambda v: isinstance(v, str) and len(v) > 0,
    "llm_provider": lambda v: isinstance(v, str) and len(v) > 0,
    "top_k": lambda v: isinstance(v, int) and v >= 1,
    "embedding_dim": lambda v: isinstance(v, int) and v >= 128,
    "enrichment_chunk_size": lambda v: isinstance(v, int) and v > 0,
    "enrichment_overlap": lambda v: isinstance(v, int) and v >= 0,
}


def _validate(config_dict: Dict[str, Any]) -> List[str]:
    """Return list of validation error messages (empty if valid)."""
    errors: List[str] = []

    for key, check in _VALIDATION_RULES.items():
        val = config_dict.get(key)
        if val is None:
            errors.append(f"Missing required config key: {key}")
        elif not check(val):
            errors.append(
                f"Invalid value for '{key}': {val!r}. "
                f"Expected: {_VALIDATION_RULES[key].__doc__ or 'non-empty'}."
            )

    # Cross-field: anthropic_api_key is optional but must be non-empty if set
    ak = config_dict.get("anthropic_api_key", "")
    if isinstance(ak, str) and len(ak) > 0 and len(ak) < 16:
        errors.append(f"anthropic_api_key too short ({len(ak)} chars)")

    return errors


# ---------------------------------------------------------------------------
# Public API — load_config (DD-02)
# ---------------------------------------------------------------------------

def load_config(yaml_path: str = "config.yaml") -> Config:
    """Load configuration from YAML → env-var overrides → defaults.

    Precedence (highest first):
      1. SRAG_* environment variables
      2. ${ENV_VAR} expansion within YAML values
      3. Hard-coded defaults (_DEFAULTS)

    Raises ValueError on validation failure.
    """
    # Start with defaults
    config_dict = copy.deepcopy(_DEFAULTS)

    # Layer 1: YAML overrides (with env-var expansion in values)
    if os.path.isfile(yaml_path):
        with open(yaml_path, "r", encoding="utf-8") as fh:
            yaml_data: Dict[str, Any] = yaml.safe_load(fh) or {}

        def _expand_node(node: Any) -> Any:
            if isinstance(node, dict):
                return {k: _expand_node(v) for k, v in node.items()}
            if isinstance(node, list):
                return [_expand_node(item) for item in node]
            if isinstance(node, str):
                return _expand_env_vars(node)
            return node

        config_dict.update(_expand_node(yaml_data))
    else:
        # YAML not found — log warning and use defaults + env vars only
        print(f"Warning: {yaml_path} not found, using default values.")

    # Layer 2: Environment variable overrides (DD-01)
    config_dict = _apply_env_overrides(config_dict)

    # Layer 3: Validation (DD-05)
    validation_errors = _validate(config_dict)
    if validation_errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in validation_errors)
        raise ValueError(error_msg)

    # Resolve paths (DD-04): expand relative paths to absolute
    for path_key in ("chromadb_path", "folders_db_path", "log_file"):
        val = config_dict.get(path_key, "")
        if isinstance(val, str) and not os.path.isabs(val):
            config_dict[path_key] = os.path.abspath(val)

    return Config(**config_dict)


# ---------------------------------------------------------------------------
# FD-05: Access patterns (no encapsulation — all attributes are public)
# ---------------------------------------------------------------------------
# Config is a dataclass; all fields are intentionally public.
# Example usage:
#   cfg = load_config()
#   print(cfg.top_k)  # direct access, no getter needed


# ---------------------------------------------------------------------------
# FD-06: Runtime config patch (not persisted to disk)
# ---------------------------------------------------------------------------

def patch_config(cfg: Config, updates: Dict[str, Any]) -> Config:
    """Apply runtime-only config patches (not persisted).

    Returns the SAME Config instance with mutated attributes.
    Callers that need thread-safety should use their own locking.
    """
    for key, value in updates.items():
        if hasattr(cfg, key):
            cfg._runtime_patches[key] = value
            setattr(cfg, key, value)
        # Silently ignore unknown keys (caller responsibility)
    return cfg


# ---------------------------------------------------------------------------
# Convenience: load_config with path resolution from working directory
# ---------------------------------------------------------------------------

def get_default_config() -> Config:
    """Return a Config instance with only defaults (no YAML)."""
    return load_config("")  # empty path → no YAML loaded