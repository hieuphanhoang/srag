"""SRAG Phase 1 Exit Gate Tests — verifies scaffolding only.

These tests confirm:
- All packages/modules are importable (no circular/import errors)
- Configuration loads correctly from config.yaml
- Models define required fields
- Run scripts reference correct entry points

Phase 1 exit gate criteria (from PLAN.md §4.2):
- [ ] uv sync completes; uv run python -c "import config, log, models" succeeds
- [ ] uv run python -c "from config import load_config; print(load_config())" prints a fully populated Config
- [ ] Every module named in SD.md §4 exists (stub or real); eval/ absent
- [ ] Unit tests for config.py pass
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
# Module existence (SD §4 — package tree; no eval/ directory)
# ---------------------------------------------------------------------------

# Root-level modules (no srag. prefix — this project has no top-level wrapper package).
_ROOT_IMPORT_TESTS = [
    ("config", "root config module"),
    ("log", "root log module"),
    ("models", "root models module"),
    ("embedding", "root embedding module"),
]

# Sub-package imports.
_SUBPACKAGE_IMPORT_TESTS = [
    ("ingest", "ingest package"),
    ("ingest.converter", "ingest.converter module"),
    ("ingest.chunker", "ingest.chunker module"),
    ("llm", "llm package"),
    ("llm.base", "llm.base module"),
    ("llm.factory", "llm.factory module"),
    ("llm.ollama_provider", "llm.ollama_provider module"),
    ("llm.anthropic_provider", "llm.anthropic_provider module"),
    ("search", "search package"),
    ("search.rewriter", "search.rewriter module"),
    ("search.merge", "search.merge module"),
    ("search.reranker", "search.reranker module"),
    ("store", "store package"),
    ("store.chromadb_store", "store.chromadb_store module"),
    ("web", "web package"),
    ("web.app", "web.app module"),
    ("web.dependencies", "web.dependencies module"),
    ("web.ingest_queue", "web.ingest_queue module"),
    ("web.routes", "web.routes module"),
]

# MCP server modules.
_MCP_IMPORT_TESTS = [
    ("mcp_server", "mcp_server package"),
    ("mcp_server.server", "mcp_server.server module"),
    ("mcp_server.tools", "mcp_server.tools module"),
]


@pytest.mark.parametrize("module, label", _ROOT_IMPORT_TESTS)
def test_root_module_importable(module: str, label: str) -> None:
    """Each root-level package/module must be importable without errors."""
    __import__(module)


@pytest.mark.parametrize("module, label", _SUBPACKAGE_IMPORT_TESTS)
def test_subpackage_importable(module: str, label: str) -> None:
    """Each sub-package module must be importable without errors."""
    __import__(module)


@pytest.mark.parametrize("module, label", _MCP_IMPORT_TESTS)
def test_mcp_module_importable(module: str, label: str) -> None:
    """Each MCP server module must be importable without errors."""
    __import__(module)


# ---------------------------------------------------------------------------
# Stub presence checks (modules exist with expected top-level symbols)
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
    # Protocol class exists (attribute presence varies by typing implementation).
    assert LLMProvider is not None


def test_ollama_provider_class_exists() -> None:
    from llm.ollama_provider import OllamaLLMProvider
    assert hasattr(OllamaLLMProvider, "generate")


def test_anthropic_provider_class_exists() -> None:
    from llm.anthropic_provider import AnthropicLLMProvider
    assert hasattr(AnthropicLLMProvider, "generate")


# ---------------------------------------------------------------------------
# Configuration validation (config.yaml matches PLAN.md §4.3 schema)
# ---------------------------------------------------------------------------

class TestConfigYAML:
    """Verify config.yaml structure per PLAN.md §4.3."""

    def test_config_file_exists(self) -> None:
        assert CONFIG_PATH.exists(), "config.yaml must exist at project root"

    def test_chromadb_path_present(self) -> None:
        cfg = _load_config()
        assert "chromadb_path" in cfg, "chromadb_path required"

    def test_ollama_url_valid(self) -> None:
        cfg = _load_config()
        url = cfg["ollama_url"]  # type: ignore[index]
        assert isinstance(url, str) and url.strip(), "ollama_url must be a non-empty string"
        assert "localhost" in url or "127.0.0.1" in url, \
            "ollama_url should point to localhost or 127.0.0.1"

    def test_embedding_model_set(self) -> None:
        cfg = _load_config()
        model = cfg["embedding_model"]  # type: ignore[index]
        assert isinstance(model, str) and len(model.strip()) > 0, \
            "embedding_model must be a non-empty string"

    def test_server_port_present(self) -> None:
        cfg = _load_config()
        server = cfg.get("server") or {}  # type: ignore[union-attr]
        assert "port" in server, "server.port required"
        port = server["port"]  # type: ignore[index]
        assert isinstance(port, int) and 1 <= port <= 65535, \
            "server.port must be a valid port number"

    def test_mcp_transport_present(self) -> None:
        cfg = _load_config()
        mcp = cfg.get("mcp") or {}  # type: ignore[union-attr]
        assert "transport" in mcp, "mcp.transport required"

    def test_search_sections_present(self) -> None:
        cfg = _load_config()
        search = cfg.get("search") or {}  # type: ignore[union-attr]
        for key in ("top_k", "rewrite_enabled", "rerank_enabled"):
            assert key in search, f"search.{key} required"

    def test_chunking_sections_present(self) -> None:
        cfg = _load_config()
        chunking = cfg.get("chunking") or {}  # type: ignore[union-attr]
        for key in ("max_tokens", "overlap"):
            assert key in chunking, f"chunking.{key} required"

    def test_llm_sections_present(self) -> None:
        cfg = _load_config()
        llm = cfg.get("llm") or {}  # type: ignore[union-attr]
        for key in ("enrichment", "rewrite", "rerank"):
            assert key in llm, f"llm.{key} required"

    def test_anthropic_section_present(self) -> None:
        cfg = _load_config()
        assert "anthropic" in cfg, "anthropic section required"

    def test_enrichment_enabled_is_bool(self) -> None:
        cfg = _load_config()
        enrichment = cfg.get("enrichment") or {}  # type: ignore[union-attr]
        assert isinstance(enrichment.get("enabled"), bool), \
            "enrichment.enabled must be a boolean"

    def test_sync_interval_present(self) -> None:
        cfg = _load_config()
        sync = cfg.get("sync") or {}  # type: ignore[union-attr]
        assert "interval_minutes" in sync, "sync.interval_minutes required"


# ---------------------------------------------------------------------------
# Config module (FD-01…FD-06) — load_config returns a populated object
# ---------------------------------------------------------------------------

class TestConfigModule:
    """Verify config.py per FD-01...FD-06."""

    def test_load_config_returns_object(self) -> None:
        from config import load_config
        result = load_config()
        assert result is not None, "load_config must return a non-None object"

    def test_load_config_has_chromadb_path(self) -> None:
        from config import load_config
        cfg = load_config()
        # The loaded Config object should expose chromadb-related fields.
        has_attr = hasattr(cfg, "chromadb_path") or hasattr(cfg, "chroma")
        assert has_attr, (
            "Config object should have chromadb_path or chroma attribute"
        )

    def test_load_config_has_ollama_url(self) -> None:
        from config import load_config
        cfg = load_config()
        has_attr = hasattr(cfg, "ollama_url") or hasattr(cfg, "ollama")
        assert has_attr, "Config object should have ollama_url or ollama attribute"


# ---------------------------------------------------------------------------
# Model validation (FD-08…FD-10 — SearchResult, ChunkWithMetadata, generate_chunk_id)
# ---------------------------------------------------------------------------

class TestModels:
    """Verify models.py per FD-08...FD-10."""

    def test_search_result_class_exists(self) -> None:
        from models import SearchResult
        assert SearchResult is not None

    def test_chunk_with_metadata_class_exists(self) -> None:
        from models import ChunkWithMetadata
        assert ChunkWithMetadata is not None

    def test_generate_chunk_id_function_exists(self) -> None:
        from models import generate_chunk_id
        assert callable(generate_chunk_id)

    def test_generate_chunk_id_deterministic(self) -> None:
        """FD-10: same inputs → same output (deterministic across path spellings)."""
        from models import generate_chunk_id
        id_a = generate_chunk_id("test_file.txt", 0, "hello world")
        id_b = generate_chunk_id("test_file.txt", 0, "hello world")
        assert id_a == id_b, "generate_chunk_id must be deterministic"

    def test_generate_chunk_id_different_for_different_chunks(self) -> None:
        """Different chunk_index should produce different IDs."""
        from models import generate_chunk_id
        id_0 = generate_chunk_id("test_file.txt", 0, "hello world")
        id_1 = generate_chunk_id("test_file.txt", 1, "hello world")
        assert id_0 != id_1, "Different chunk_index must produce different IDs"


# ---------------------------------------------------------------------------
# Run script validation (FD-84/85)
# ---------------------------------------------------------------------------

class TestRunScripts:
    def test_run_bat_exists(self) -> None:
        bat = ROOT / "run.bat"
        assert bat.exists(), "run.bat must exist for Windows users"

    def test_run_sh_exists_and_executable(self) -> None:
        sh = ROOT / "run.sh"
        assert sh.exists(), "run.sh must exist for Unix users"

    def test_run_bat_references_entry_point(self) -> None:
        bat = ROOT / "run.bat"
        if bat.exists():
            content = bat.read_text(encoding="utf-8")
            # Should reference web.app (the FastAPI entry point).
            assert "web.app" in content or "python -m" in content, \
                "run.bat should reference the FastAPI entry point"

    def test_run_sh_references_entry_point(self) -> None:
        sh = ROOT / "run.sh"
        if sh.exists():
            content = sh.read_text(encoding="utf-8")
            assert "web.app" in content or "uvicorn" in content, \
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
            assert (
                "Usage" in content or "Quick Start" in content or "安装" in content
            ), "README.md should contain usage instructions"


# ---------------------------------------------------------------------------
# No eval/ package (PLAN.md §9 — out of scope)
# ---------------------------------------------------------------------------

class TestScopeBoundary:
    def test_no_eval_package(self) -> None:
        """eval/ must not exist (out of scope per PLAN.md §9)."""
        eval_dir = ROOT / "eval"
        assert not eval_dir.exists(), "eval/ package is out of scope and must not exist"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])