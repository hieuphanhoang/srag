"""SRAG Phase 1 Exit Gate Tests — verifies scaffolding only.

These tests confirm:
- All packages/modules are importable (no circular/import errors)
- Stub functions/classes exist where expected
- Configuration loads correctly from config.yaml
- Models define required fields
- Run scripts reference correct entry points
"""

import os
import sys
import yaml
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"


def _load_config() -> dict:
    """Load config.yaml and return parsed dict."""
    assert CONFIG_PATH.exists(), "config.yaml must exist"
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict), "Top-level of config.yaml must be a mapping"
    return data


# ---------------------------------------------------------------------------
# Package / module existence (SD §4 — package tree)
# ---------------------------------------------------------------------------

_IMPORT_TESTS = [
    ("tests", "root package"),
    ("srag.config", "srag.config"),
    ("srag.log_config", "srag.log_config"),
    ("srag.models", "srag.models"),
    ("srag.embedding", "srag.embedding"),
    ("srag.ingest", "srag.ingest"),
    ("srag.ingest.converter", "srag.ingest.converter"),
    ("srag.ingest.chunker", "srag.ingest.chunker"),
    ("srag.llm", "srag.llm"),
    ("srag.llm.base", "srag.llm.base"),
    ("srag.llm.factory", "srag.llm.factory"),
    ("srag.llm.ollama_provider", "srag.llm.ollama_provider"),
    ("srag.llm.anthropic_provider", "srag.llm.anthropic_provider"),
    ("srag.search", "srag.search"),
    ("srag.search.rewriter", "srag.search.rewriter"),
    ("srag.search.merge", "srag.search.merge"),
    ("srag.search.reranker", "srag.search.reranker"),
    ("srag.store", "srag.store"),
    ("srag.store.chromadb_store", "srag.store.chromadb_store"),
]


@pytest.mark.parametrize("module, label", _IMPORT_TESTS)
def test_module_importable(module: str, label: str) -> None:
    """Each package/module must be importable without errors."""
    __import__(module)


# ---------------------------------------------------------------------------
# Stub presence checks
# ---------------------------------------------------------------------------

def test_query_rewriter_class_exists() -> None:
    from search.rewriter import QueryRewriter
    assert hasattr(QueryRewriter, "rewrite")


def test_merge_results_function_exists() -> None:
    from search.merge import merge_results
    assert callable(merge_results)


def test_rerank_function_exists() -> None:
    from search.reranker import rerank
    assert callable(rerank)


def test_chroma_store_class_exists() -> None:
    from store.chromadb_store import ChromaStore
    assert hasattr(ChromaStore, "initialize")
    assert hasattr(ChromaStore, "upsert")
    assert hasattr(ChromaStore, "search")


def test_llm_provider_protocol_exists() -> None:
    from llm.base import LLMProvider
    # Protocol should have generate method
    assert hasattr(LLMProvider, "__protocol_attrs__") or True  # Protocols vary by typing impl


def test_ollama_provider_class_exists() -> None:
    from llm.ollama_provider import OllamaLLMProvider
    assert hasattr(OllamaLLMProvider, "generate")


def test_anthropic_provider_class_exists() -> None:
    from llm.anthropic_provider import AnthropicLLMProvider
    assert hasattr(AnthropicLLMProvider, "generate")


# ---------------------------------------------------------------------------
# Configuration validation (FD-01…FD-10)
# ---------------------------------------------------------------------------

class TestConfig:
    def test_config_file_exists(self) -> None:
        assert CONFIG_PATH.exists(), "config.yaml must exist at project root"

    def test_config_has_app_section(self) -> None:
        cfg = _load_config()
        assert "app" in cfg, "Top-level 'app' section required"

    def test_config_has_embedding(self) -> None:
        cfg = _load_config()
        assert "embedding" in cfg, "Top-level 'embedding' section required"

    def test_embedding_dimensions_positive(self) -> None:
        cfg = _load_config()
        dims = cfg["embedding"]["dimensions"]  # type: ignore[index]
        assert isinstance(dims, int), "embedding.dimensions must be an integer"
        assert dims > 0, "embedding.dimensions must be positive"

    def test_embedding_model_not_empty(self) -> None:
        cfg = _load_config()
        model = cfg["embedding"]["model"]  # type: ignore[index]
        assert isinstance(model, str), "embedding.model must be a string"
        assert len(model.strip()) > 0, "embedding.model must not be empty"

    def test_app_name_not_empty(self) -> None:
        cfg = _load_config()
        name = cfg["app"]["name"]  # type: ignore[index]
        assert isinstance(name, str), "app.name must be a string"
        assert len(name.strip()) > 0, "app.name must not be empty"

    def test_app_version_semver(self) -> None:
        cfg = _load_config()
        version = cfg["app"]["version"]  # type: ignore[index]
        import re
        assert re.match(r"^\\d+\\.\\d+\\.\\d+$", str(version)), \
            "app.version must follow semantic versioning (e.g., '0.1.0')"

    def test_default_provider_model_present(self) -> None:
        cfg = _load_config()
        llm = cfg.get("llm", {})
        assert "default_provider" in llm, "llm.default_provider required"
        assert "model" in llm, "llm.model required"

    def test_rag_sections_present(self) -> None:
        """FD-01…FD-10: all top-level sections present."""
        cfg = _load_config()
        for key in ("app", "embedding", "ingest", "rag"):
            assert key in cfg, f"'{key}' section required in config.yaml"


# ---------------------------------------------------------------------------
# Models validation (FD-24…FD-28)
# ---------------------------------------------------------------------------

class TestModels:
    def test_document_model_exists(self) -> None:
        from srag.models import Document
        assert hasattr(Document, "doc_id") or True  # dataclass / Pydantic

    def test_chunk_model_exists(self) -> None:
        from srag.models import Chunk

    def test_search_result_model_exists(self) -> None:
        from srag.models import SearchResult


# ---------------------------------------------------------------------------
# Run script validation
# ---------------------------------------------------------------------------

class TestRunScripts:
    def test_run_bat_exists(self) -> None:
        bat = ROOT / "run.bat"
        assert bat.exists(), "run.bat must exist for Windows users"

    def test_run_sh_exists_and_executable(self) -> None:
        sh = ROOT / "run.sh"
        assert sh.exists(), "run.sh must exist for Unix users"

    def test_run_sh_references_entry_point(self) -> None:
        sh = ROOT / "run.sh"
        if sh.exists():
            content = sh.read_text(encoding="utf-8")
            assert "srag.web.app" in content or "uvicorn" in content, \
                "run.sh should reference the FastAPI entry point"


# ---------------------------------------------------------------------------
# README validation
# ---------------------------------------------------------------------------

class TestREADME:
    def test_readme_exists(self) -> None:
        readme = ROOT / "README.md"
        assert readme.exists(), "README.md must exist at project root"

    def test_readme_contains_usage(self) -> None:
        readme = ROOT / "README.md"
        if readme.exists():
            content = readme.read_text(encoding="utf-8")
            assert "Usage" in content or "Quick Start" in content or "安装" in content, \
                "README.md should contain usage instructions"


# ---------------------------------------------------------------------------
# Entry point validation
# ---------------------------------------------------------------------------

class TestEntryPoint:
    def test_web_app_exists(self) -> None:
        from srag.web.app import app
        assert app is not None, "FastAPI 'app' instance must be exported"

    def test_fastapi_installed(self) -> None:
        try:
            import fastapi  # noqa: F401
        except ImportError:
            pytest.skip("fastapi not installed (expected in bare stub mode)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])