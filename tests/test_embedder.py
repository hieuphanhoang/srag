"""Unit tests for embedding.py — covers Embedder class behavior.

Key coverage:
- Embedder initialization with different parameters
- embed() method (single text embedding)
- embed_batch() method (batch text embedding)
- dimension property (inference from model response)
- Error handling (missing ollama package, empty input)
"""

import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_ollama_cache():
    """Safety net around embedding._get_ollama()'s module-level cache.

    Several tests below mock the ollama import via `sys.modules` or
    `_get_ollama` directly, and (this file has twice shipped a test that
    forgot to) need to restore `embedding._ollama` afterward. If any test
    leaves a MagicMock cached there, every *later* real Embedder.embed()
    call in the process - including in a completely different test file,
    e.g. tests/test_integration.py - silently returns fake data instead of
    hitting Ollama. Autouse makes this reset unconditional instead of
    depending on each test remembering to do it.

    Also resets embedding.Embedder's per-instance client cache is moot
    here since each test builds a fresh Embedder(); nothing to reset there.
    """
    import embedding as emb_module
    orig = emb_module._ollama
    emb_module._ollama = None
    yield
    emb_module._ollama = orig


def _mock_ollama_module(embeddings_response: dict[str, Any] | None = None, side_effect=None) -> MagicMock:
    """Build a fake `ollama` module whose Client(host=...).embed(...) is mocked.

    Embedder builds `ollama.Client(host=self._url)` and calls `.embed(...)`
    on that client instance (not the bare `ollama.embed()` module function,
    which has no host parameter at all and would silently ignore any
    configured URL) - so the mock needs to sit at `mock_module.Client(...).embed`.
    """
    mock_module = MagicMock()
    mock_client = MagicMock()
    if side_effect is not None:
        mock_client.embed = MagicMock(side_effect=side_effect)
    else:
        mock_client.embed = MagicMock(return_value=embeddings_response)
    mock_module.Client = MagicMock(return_value=mock_client)
    return mock_module


@pytest.fixture
def mock_ollama_module() -> MagicMock:
    """Fixture variant of _mock_ollama_module(), patched into sys.modules.

    Yields the fake ollama *module*; tests configure the response via
    `mock_ollama_module.Client.return_value.embed = MagicMock(...)`.
    """
    mock = _mock_ollama_module()
    with patch.dict(sys.modules, {'ollama': mock}):
        yield mock


# ---------------------------------------------------------------------------
# Embedder initialization tests
# ---------------------------------------------------------------------------

class TestEmbedderInit:
    """Test Embedder.__init__ with different parameters."""

    def test_default_init(self):
        from embedding import Embedder
        e = Embedder()
        assert e._url == "http://localhost:11434"
        assert e._model == "qwen3-embedding"
        assert e._dimension is None

    def test_init_with_custom_url(self):
        from embedding import Embedder
        e = Embedder(url="http://custom:8080")
        assert e._url == "http://custom:8080"

    def test_init_with_custom_model(self):
        from embedding import Embedder
        e = Embedder(model="my-embed-model")
        assert e._model == "my-embed-model"

    def test_init_with_dimension(self):
        from embedding import Embedder
        e = Embedder(dimension=512)
        assert e._dimension == 512


# ---------------------------------------------------------------------------
# embed() method tests
# ---------------------------------------------------------------------------

class TestEmbedderEmbed:
    """Test embed() single text embedding."""

    def test_embed_returns_list_of_floats(self, mock_ollama_module):
        from embedding import Embedder
        e = Embedder()

        default_embedding = [0.01] * 1024
        mock_ollama_module.Client.return_value.embed = MagicMock(
            return_value={"embeddings": [default_embedding]}
        )

        result = e.embed("Hello world")
        assert isinstance(result, list)
        assert all(isinstance(x, float) for x in result)

    def test_embed_returns_correct_length(self, mock_ollama_module):
        from embedding import Embedder
        e = Embedder()

        default_embedding = [0.01] * 1024
        mock_ollama_module.Client.return_value.embed = MagicMock(
            return_value={"embeddings": [default_embedding]}
        )

        result = e.embed("Test input")
        assert len(result) == 1024

    def test_embed_with_truncated_dimension(self):
        from embedding import Embedder

        default_embedding = [0.01] * 1024
        mock_mod = _mock_ollama_module({"embeddings": [default_embedding]})

        with patch.dict(sys.modules, {'ollama': mock_mod}):
            e = Embedder(dimension=512)
            result = e.embed("Test")
            assert len(result) == 512

    def test_embed_uses_configured_url(self):
        """Regression test: Embedder previously called the bare
        ollama.embed() module function, which has no host parameter at
        all - it always hit ollama's default client regardless of the
        configured url. Assert Client(host=...) is actually built with the
        Embedder's url.
        """
        from embedding import Embedder
        import embedding as emb_module

        default_embedding = [0.01] * 1024
        mock_mod = _mock_ollama_module({"embeddings": [default_embedding]})

        with patch.object(emb_module, '_get_ollama', return_value=mock_mod):
            e = Embedder(model="test-model", url="http://custom-host:9999")
            e.embed("Test text")

            mock_mod.Client.assert_called_once_with(host="http://custom-host:9999")
            mock_mod.Client.return_value.embed.assert_called_once()
            call_args = mock_mod.Client.return_value.embed.call_args
            assert call_args.kwargs['model'] == "test-model"


# ---------------------------------------------------------------------------
# embed_batch() method tests
# ---------------------------------------------------------------------------

class TestEmbedderEmbedBatch:
    """Test embed_batch() multiple text embedding."""

    def test_embed_batch_empty_input(self):
        from embedding import Embedder
        e = Embedder()

        # Without mocking - should return empty list immediately
        result = e.embed_batch([])
        assert result == []

    def test_embed_batch_single_item(self, mock_ollama_module):
        from embedding import Embedder

        default_embedding = [0.01] * 1024
        mock_ollama_module.Client.return_value.embed = MagicMock(
            return_value={"embeddings": [default_embedding]}
        )

        e = Embedder()
        result = e.embed_batch(["Single text"])
        assert len(result) == 1
        assert isinstance(result[0], list)

    def test_embed_batch_multiple_items(self):
        from embedding import Embedder
        import embedding as emb_module

        default_embedding = [0.01] * 1024
        mock_mod = _mock_ollama_module({"embeddings": [default_embedding, default_embedding]})

        with patch.object(emb_module, '_get_ollama', return_value=mock_mod):
            e = Embedder()
            result = e.embed_batch(["Text 1", "Text 2"])
            assert len(result) == 2

    def test_embed_batch_truncates_to_dimension(self):
        from embedding import Embedder

        default_embedding = [0.01] * 1024
        mock_mod = _mock_ollama_module(
            {"embeddings": [list(default_embedding), list(default_embedding)]}
        )

        with patch.dict(sys.modules, {'ollama': mock_mod}):
            e = Embedder(dimension=256)
            result = e.embed_batch(["Text 1", "Text 2"])
            assert all(len(r) == 256 for r in result)


# ---------------------------------------------------------------------------
# dimension property tests
# ---------------------------------------------------------------------------

class TestEmbedderDimension:
    """Test dimension property inference."""

    def test_dimension_inferred_on_first_call(self):
        from embedding import Embedder
        import embedding as emb_module

        default_embedding = [0.01] * 512
        mock_mod = _mock_ollama_module({"embeddings": [default_embedding]})

        with patch.object(emb_module, '_get_ollama', return_value=mock_mod):
            e = Embedder()
            dim = e.dimension
            assert dim == 512

    def test_dimension_cached_after_inference(self):
        from embedding import Embedder
        import embedding as emb_module

        default_embedding = [0.01] * 768
        mock_mod = _mock_ollama_module({"embeddings": [default_embedding]})

        with patch.object(emb_module, '_get_ollama', return_value=mock_mod):
            e = Embedder()
            # First call triggers inference
            dim1 = e.dimension
            assert e._dimension == 768

            # Second call should use cached value (no additional embed calls)
            dim2 = e.dimension
            assert dim2 == 768


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------

class TestEmbedderErrorHandling:
    """Test error handling scenarios."""

    def test_ollama_import_error(self):
        """Test RuntimeError when ollama package is not installed."""
        import embedding as emb_module

        # Save the real _get_ollama behavior by patching its internal import
        with patch.object(emb_module, '_get_ollama') as mock_get_ollama:
            mock_get_ollama.side_effect = RuntimeError("ollama module not found. Please install ollama package.")

            # Now test that calling _get_ollama raises RuntimeError
            with pytest.raises(RuntimeError, match="ollama"):
                mock_get_ollama()

    def test_embed_batch_returns_empty_for_empty_input(self):
        """Test that empty input returns empty output."""
        from embedding import Embedder
        e = Embedder()

        # When no ollama, embed_batch([]) should return [] early
        result = e.embed_batch([])
        assert result == []


# ---------------------------------------------------------------------------
# Integration-style tests (without actual Ollama server)
# ---------------------------------------------------------------------------

class TestEmbedderMockedIntegration:
    """Test realistic usage patterns with mocks."""

    def test_full_workflow(self):
        """Test a typical embedding workflow."""
        from embedding import Embedder
        import embedding as emb_module

        # Different embeddings for different inputs (simulating real behavior)
        def embed_side_effect(**kwargs):
            input_val = kwargs.get('input', '')
            if isinstance(input_val, str):
                # Single text - return 1024-dim embedding
                embedding = [hash(input_val + str(i)) % 100 / 10000 for i in range(1024)]
            else:
                # Batch - return one per input
                embedding = [[hash(t + str(i)) % 100 / 10000 for i in range(1024)] for t in input_val]
            return {"embeddings": embedding if isinstance(input_val, list) else [embedding]}

        mock_mod = _mock_ollama_module(side_effect=embed_side_effect)

        with patch.object(emb_module, '_get_ollama', return_value=mock_mod):
            e = Embedder(model="qwen3-embedding")

            # Single embed
            vec1 = e.embed("First document")
            assert len(vec1) == 1024

            # Batch embed
            vectors = e.embed_batch(["Doc A", "Doc B"])
            assert len(vectors) == 2
            for v in vectors:
                assert len(v) == 1024

    def test_dimension_property_workflow(self):
        """Test dimension property with realistic flow."""
        from embedding import Embedder
        import embedding as emb_module

        default_embedding = [0.01] * 384
        mock_mod = _mock_ollama_module({"embeddings": [default_embedding]})

        with patch.object(emb_module, '_get_ollama', return_value=mock_mod):
            e = Embedder()

            # Before calling dimension, _dimension is None
            assert e._dimension is None

            # First access infers dimension
            dim = e.dimension
            assert dim == 384
            assert e._dimension == 384


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
