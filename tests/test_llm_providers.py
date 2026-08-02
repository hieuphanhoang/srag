"""Tests for llm.factory, llm.ollama_provider, llm.anthropic_provider (Phase 3)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from llm.anthropic_provider import AnthropicProvider
from llm.base import GenerationResult
from llm.factory import create_client, create_llm_provider
from llm.ollama_provider import OllamaProvider


# ─── factory.create_llm_provider ────────────────────────────────────

class TestCreateLlmProvider:
    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError):
            create_llm_provider("not-a-provider")

    def test_ollama_creates_ollama_provider(self):
        provider = create_llm_provider("ollama", model="llama3.2:latest")
        assert isinstance(provider, OllamaProvider)
        assert provider.model_name == "llama3.2:latest"

    def test_provider_name_case_insensitive(self):
        provider = create_llm_provider("OLLAMA", model="m")
        assert isinstance(provider, OllamaProvider)

    def test_anthropic_creates_anthropic_provider(self):
        provider = create_llm_provider("anthropic", model="claude-sonnet-5", api_key="sk-ant-test-key-value")
        assert isinstance(provider, AnthropicProvider)
        assert provider.model_name == "claude-sonnet-5"


# ─── factory.create_client ──────────────────────────────────────────

class TestCreateClient:
    def test_invalid_spec_missing_slash_raises(self):
        with pytest.raises(ValueError):
            create_client("no-slash-here")

    def test_none_spec_returns_default_ollama_provider(self):
        provider = create_client(None)
        assert isinstance(provider, OllamaProvider)

    def test_ollama_spec_pulls_base_url_from_config(self):
        fake_cfg = SimpleNamespace(ollama_url="http://custom-host:11434", anthropic_api_key="")
        with patch("config.load_config", return_value=fake_cfg):
            with patch("llm.ollama_provider.OllamaProvider.__init__", return_value=None) as init:
                create_client("ollama/some-model")
                _, kwargs = init.call_args
                assert kwargs["base_url"] == "http://custom-host:11434"

    def test_anthropic_spec_pulls_api_key_from_config(self):
        fake_cfg = SimpleNamespace(ollama_url="http://localhost:11434", anthropic_api_key="sk-ant-from-config")
        with patch("config.load_config", return_value=fake_cfg):
            with patch("llm.anthropic_provider.AnthropicProvider.__init__", return_value=None) as init:
                create_client("anthropic/claude-sonnet-5")
                _, kwargs = init.call_args
                assert kwargs["api_key"] == "sk-ant-from-config"

    def test_config_load_failure_falls_back_gracefully(self):
        with patch("config.load_config", side_effect=RuntimeError("no config")):
            provider = create_client("ollama/some-model")
            assert isinstance(provider, OllamaProvider)


# ─── OllamaProvider ──────────────────────────────────────────────────

class TestOllamaProvider:
    def test_empty_prompt_raises(self):
        provider = OllamaProvider(model="m")
        with pytest.raises(ValueError):
            provider.generate("")

    def test_generate_success(self):
        provider = OllamaProvider(model="m")
        provider._client = MagicMock()
        provider._client.chat.return_value = {
            "message": {"content": "hello there"},
            "prompt_eval_count": 10,
            "eval_count": 5,
        }
        result = provider.generate("hi", max_tokens=20)
        assert isinstance(result, GenerationResult)
        assert result.text == "hello there"
        assert result.model == "m"
        assert result.usage == {"prompt_tokens": 10, "completion_tokens": 5}

    def test_generate_wraps_client_errors(self):
        provider = OllamaProvider(model="m")
        provider._client = MagicMock()
        provider._client.chat.side_effect = RuntimeError("connection refused")
        with pytest.raises(RuntimeError):
            provider.generate("hi")

    def test_generate_batch_isolates_failures(self):
        provider = OllamaProvider(model="m")
        provider._client = MagicMock()

        def chat_side_effect(*args, **kwargs):
            content = kwargs.get("messages", [{}])[0].get("content", "")
            if content == "bad":
                raise RuntimeError("boom")
            return {"message": {"content": f"ok:{content}"}}

        provider._client.chat.side_effect = chat_side_effect
        results = provider.generate_batch(["good", "bad"])
        assert results[0].text == "ok:good"
        assert results[1].text == ""

    def test_model_name_property(self):
        provider = OllamaProvider(model="qwen3:4b")
        assert provider.model_name == "qwen3:4b"


@pytest.mark.integration
class TestOllamaProviderLive:
    def test_generate_against_real_ollama(self):
        """Requires a running local Ollama with at least one pulled model."""
        import ollama as _ollama

        client = _ollama.Client(host="http://localhost:11434")
        try:
            models = client.list().get("models", [])
        except Exception:
            pytest.skip("Ollama not reachable on localhost:11434")
        if not models:
            pytest.skip("No local Ollama models available")

        model_name = models[0].model
        provider = OllamaProvider(model=model_name)
        result = provider.generate("Reply with exactly: OK", max_tokens=10)
        assert isinstance(result.text, str)


# ─── AnthropicProvider ───────────────────────────────────────────────

class TestAnthropicProvider:
    def test_empty_prompt_raises(self):
        provider = AnthropicProvider(model="claude-sonnet-5", api_key="sk-ant-test-key")
        with pytest.raises(ValueError):
            provider.generate("")

    def test_generate_success(self):
        provider = AnthropicProvider(model="claude-sonnet-5", api_key="sk-ant-test-key")
        provider._client = MagicMock()
        provider._client.api_key = "sk-ant-test-key"
        block = SimpleNamespace(text="hello from claude")
        usage = SimpleNamespace(input_tokens=12, output_tokens=8)
        provider._client.messages.create.return_value = SimpleNamespace(content=[block], usage=usage)

        result = provider.generate("hi")
        assert result.text == "hello from claude"
        assert result.model == "claude-sonnet-5"
        assert result.usage == {"input_tokens": 12, "output_tokens": 8}

    def test_missing_api_key_raises(self):
        provider = AnthropicProvider(model="claude-sonnet-5", api_key="sk-ant-test-key")
        provider._client = MagicMock()
        provider._client.api_key = None
        with pytest.raises(ValueError):
            provider.generate("hi")

    def test_generate_wraps_client_errors(self):
        provider = AnthropicProvider(model="claude-sonnet-5", api_key="sk-ant-test-key")
        provider._client = MagicMock()
        provider._client.api_key = "sk-ant-test-key"
        provider._client.messages.create.side_effect = RuntimeError("api down")
        with pytest.raises(RuntimeError):
            provider.generate("hi")

    def test_generate_batch_isolates_failures(self):
        provider = AnthropicProvider(model="claude-sonnet-5", api_key="sk-ant-test-key")
        provider._client = MagicMock()
        provider._client.api_key = "sk-ant-test-key"

        def create_side_effect(*args, **kwargs):
            msg = kwargs["messages"][0]["content"]
            if msg == "bad":
                raise RuntimeError("boom")
            block = SimpleNamespace(text=f"ok:{msg}")
            return SimpleNamespace(content=[block], usage=SimpleNamespace(input_tokens=1, output_tokens=1))

        provider._client.messages.create.side_effect = create_side_effect
        results = provider.generate_batch(["good", "bad"])
        assert results[0].text == "ok:good"
        assert results[1].text == ""

    def test_model_name_property(self):
        provider = AnthropicProvider(model="claude-opus-5", api_key="sk-ant-test-key")
        assert provider.model_name == "claude-opus-5"
