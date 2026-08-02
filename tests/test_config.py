"""Tests for config.py — covers FD-05, DD-01..DD-06."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

# Import directly from the module (no srag package prefix needed)
from config import (
    Config,
    _apply_env_overrides,
    _expand_env_vars,
    _validate,
    get_default_config,
    load_config,
    patch_config,
)


# ------------------------------------------------------------------ #
# Helper: create a temporary YAML file with the given dict content   #
# ------------------------------------------------------------------ #
def _write_yaml(data: dict) -> str:
    """Write *data* to a temp .yaml and return its path."""
    fh = tempfile.NamedTemporaryFile(
        suffix=".yaml", mode="w", delete=False, encoding="utf-8"
    )
    try:
        yaml.dump(data, fh)
        fh.close()
    except Exception:
        fh.close()
        raise
    return fh.name


# ========================= load_config tests ======================== #

class TestConfigLoad:
    def test_ollama_url_from_yaml(self):
        """YAML ollama_url overrides the default."""
        path = _write_yaml({"ollama_url": "http://myhost:11434"})
        cfg = load_config(path)
        assert cfg.ollama_url == "http://myhost:11434"

    def test_embedding_model_from_yaml(self):
        """YAML embedding_model overrides the default."""
        path = _write_yaml({"embedding_model": "bge-large"})
        cfg = load_config(path)
        assert cfg.embedding_model == "bge-large"


class TestConfigDefaults:
    @patch("config.os.path.isfile", return_value=False)
    def test_default_ollama_url(self, _mock):
        """When no YAML is loaded the default ollama_url is used."""
        cfg = load_config("")
        assert cfg.ollama_url == "http://localhost:11434"

    @patch("config.os.path.isfile", return_value=False)
    def test_default_embedding_model(self, _mock):
        """When no YAML is loaded the default embedding_model is used."""
        cfg = load_config("")
        assert cfg.embedding_model == "qwen3-embedding"

    @patch("config.os.path.isfile", return_value=False)
    def test_default_top_k(self, _mock):
        cfg = load_config("")
        assert cfg.top_k == 10

    @patch("config.os.path.isfile", return_value=False)
    def test_default_embedding_dim(self, _mock):
        cfg = load_config("")
        assert cfg.embedding_dim == 4096


class TestConfigMissingRequired:
    def test_missing_ollama_url_raises(self):
        """Empty string ollama_url should fail validation."""
        path = _write_yaml({"ollama_url": ""})
        with pytest.raises(ValueError, match="ollama_url"):
            load_config(path)

    def test_invalid_top_k_raises(self):
        """top_k < 1 is invalid."""
        path = _write_yaml({"top_k": 0})
        with pytest.raises(ValueError, match="top_k"):
            load_config(path)

    def test_invalid_embedding_dim_raises(self):
        """embedding_dim < 128 is invalid."""
        path = _write_yaml({"embedding_dim": 64})
        with pytest.raises(ValueError, match="embedding_dim"):
            load_config(path)


class TestConfigEnvOverride:
    @patch.dict(os.environ, {"SRAG_TOP_K": "42"}, clear=False)
    def test_override_top_k(self):
        """SRAG_TOP_K env var overrides YAML value."""
        path = _write_yaml({"top_k": 5})
        cfg = load_config(path)
        assert cfg.top_k == 42

    @patch.dict(os.environ, {"SRAG_EMBEDDING_DIM": "8192"}, clear=False)
    def test_override_embedding_dim(self):
        """SRAG_EMBEDDING_DIM env var overrides YAML value."""
        path = _write_yaml({"embedding_dim": 4096})
        cfg = load_config(path)
        assert cfg.embedding_dim == 8192


class TestConfigEnvExpansion:
    def test_expand_env_vars_string(self):
        """$VAR syntax should expand to the env var value."""
        with patch.dict(os.environ, {"MY_TEST_DIR": "/tmp/test"}, clear=False):
            result = _expand_env_vars("$MY_TEST_DIR/data")
            assert result == "/tmp/test/data"

    def test_expand_env_vars_short_form(self):
        """${VAR} syntax should expand to the env var value."""
        with patch.dict(os.environ, {"MY_TEST_VAR": "hello"}, clear=False):
            result = _expand_env_vars("${MY_TEST_VAR}")
            assert result == "hello"

    def test_expand_non_string_unchanged(self):
        """Non-string values should pass through unchanged."""
        assert _expand_env_vars(123) == 123
        assert _expand_env_vars([1, 2]) == [1, 2]


class TestConfigMissingFile:
    def test_missing_file_warns_and_uses_defaults(self, capfd):
        """load_config('nonexistent.yaml') should use defaults."""
        cfg = load_config("this_file_does_not_exist_12345.yaml")
        assert cfg.ollama_url == "http://localhost:11434"


class TestPatchConfig:
    def test_patch_applies_updates(self):
        """patch_config should mutate the Config instance."""
        cfg = get_default_config()
        patch_config(cfg, {"top_k": 50})
        assert cfg.top_k == 50

    def test_patch_stored_in_runtime_patches(self):
        """Patched keys are recorded in _runtime_patches."""
        cfg = get_default_config()
        patch_config(cfg, {"top_k": 99})
        assert "top_k" in cfg._runtime_patches


class TestGetDefaultConfig:
    def test_returns_config(self):
        cfg = get_default_config()
        assert isinstance(cfg, Config)


class TestValidationRules:
    def test_top_k_validation(self):
        """top_k must be >= 1."""
        errors = _validate({"top_k": 0})
        assert any("top_k" in e for e in errors)

    def test_embedding_dim_validation(self):
        """embedding_dim must be >= 128."""
        errors = _validate({"embedding_dim": 64})
        assert any("embedding_dim" in e for e in errors)


class TestApplyEnvOverrides:
    @patch.dict(os.environ, {"SRAG_TOP_K": "7"}, clear=False)
    def test_env_override_top_k(self):
        result = _apply_env_overrides({"top_k": 10})
        assert result["top_k"] == 7

    def test_no_env_vars_set(self):
        """When no relevant env vars are set, dict is unchanged."""
        with patch.dict(os.environ, {}, clear=False):
            data = {"top_k": 5, "ollama_url": "http://x:1"}
            result = _apply_env_overrides(data)
            assert result["top_k"] == 5
            assert result["ollama_url"] == "http://x:1"


class TestValidate:
    def test_missing_ollama_url(self):
        errors = _validate({})
        assert any("ollama_url" in e for e in errors)

    def test_valid_config_no_errors(self):
        data = {
            "ollama_url": "http://localhost:11434",
            "embedding_model": "qwen3-embedding",
            "llm_provider": "ollama/qwen3:4b",
            "top_k": 10,
            "embedding_dim": 4096,
            "enrichment_chunk_size": 768,
            "enrichment_overlap": 64,
        }
        errors = _validate(data)
        assert len(errors) == 0

    def test_anthropic_key_too_short(self):
        errors = _validate({"anthropic_api_key": "abc"})
        assert any("anthropic_api_key" in e for e in errors)