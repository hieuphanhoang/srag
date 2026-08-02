"""SRAG Web Dependency Injection Container.

Provides singleton instances of all services and a lifespan context manager
that handles startup/shutdown sequencing. All web-layer modules import from
here rather than constructing objects directly.

This wires the web layer to the REAL config/ingest/search modules — config.py
(not a hand-rolled YAML loader), the SQLite FolderRegistry, and ChromaStore's
multi-collection (one collection per folder) search, matching the design in
SD.md / DD.md.
"""

from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI

from core.config import Config, load_config
from core.embedding import Embedder, get_embedder as _build_embedder
from ingest.converter import ConverterManager
from ingest.folder_registry import FolderRegistry, collection_name_for_path
from store.chromadb_store import ChromaStore, SearchResult

logger = logging.getLogger(__name__)

# ─── Global service singletons (populated lazily / by lifespan) ──────

_config: Config | None = None
_embedder: Embedder | None = None
_store: ChromaStore | None = None
_registry: FolderRegistry | None = None
_converter: ConverterManager | None = None
_reranker_cache: dict[str, Any] = {}
_lock = threading.Lock()


def get_config() -> Config:
    """Return the cached application :class:`Config`."""
    global _config
    if _config is None:
        with _lock:
            if _config is None:
                _config = load_config()
    return _config


def get_embedder() -> Embedder:
    """Return the singleton embedding client, configured from ``config.yaml``."""
    global _embedder
    if _embedder is None:
        with _lock:
            if _embedder is None:
                _embedder = _build_embedder()
    return _embedder


def get_store() -> ChromaStore:
    """Return the singleton ChromaDB store (multi-collection, one per folder)."""
    global _store
    if _store is None:
        with _lock:
            if _store is None:
                cfg = get_config()
                _store = ChromaStore(
                    persist_directory=cfg.chromadb_path,
                    tenant=cfg.chromadb_tenant,
                    database=cfg.chromadb_database,
                )
                _store.initialize()
    return _store


def get_registry() -> FolderRegistry:
    """Return the singleton SQLite-backed folder registry."""
    global _registry
    if _registry is None:
        with _lock:
            if _registry is None:
                cfg = get_config()
                _registry = FolderRegistry(cfg.folders_db_path)
    return _registry


def get_converter() -> ConverterManager:
    """Return the singleton multi-format document converter."""
    global _converter
    if _converter is None:
        with _lock:
            if _converter is None:
                _converter = ConverterManager()
    return _converter


def get_reranker():
    """Return a Reranker instance, or ``None`` if reranking is disabled/unavailable."""
    cfg = get_config()
    if not cfg.search_rerank_enabled:
        return None
    if "reranker" not in _reranker_cache:
        try:
            from search.reranker import Reranker
            _reranker_cache["reranker"] = Reranker(cfg.llm_rerank)
        except Exception as exc:
            logger.warning("Reranker unavailable, disabling rerank: %s", exc)
            _reranker_cache["reranker"] = None
    return _reranker_cache["reranker"]


def get_rewriter():
    """Return a QueryRewriter instance, or ``None`` if rewriting is disabled/unavailable."""
    cfg = get_config()
    if not cfg.search_rewrite_enabled:
        return None
    if "rewriter" not in _reranker_cache:
        try:
            from search.rewriter import QueryRewriter
            _reranker_cache["rewriter"] = QueryRewriter(cfg.llm_rewrite)
        except Exception as exc:
            logger.warning("Query rewriter unavailable, disabling rewrite: %s", exc)
            _reranker_cache["rewriter"] = None
    return _reranker_cache["rewriter"]


# ─── Folder <-> collection helpers ────────────────────────────────────

def get_or_register_folder(path: str) -> tuple[int, str]:
    """Resolve *path* to a registered folder, registering it if new.

    The ingest pipeline operates per-folder (FolderRegistry + IngestQueue),
    not per-file. Accepts either a directory (used directly) or a file
    (its parent directory is registered/ingested instead). Returns
    ``(folder_id, folder_path)``.
    """
    from ingest.folder_registry import FolderConfig

    p = Path(path).resolve()
    folder_path = str(p if p.is_dir() else p.parent)

    registry = get_registry()
    existing = registry.get_folder_by_path(folder_path)
    if existing is not None:
        return existing[0], folder_path

    folder_id = registry.add_folder(FolderConfig(path=folder_path))
    return folder_id, folder_path


def collection_for_folder(folder_path: str) -> str:
    """Deterministic ChromaDB collection name for a folder path (DD-05)."""
    return collection_name_for_path(folder_path)


def searchable_collection_names(registry: FolderRegistry) -> list[str] | None:
    """Resolve which collections a search should scope to.

    Returns ``None`` to mean "search every collection that exists" (the
    common case — this also picks up any collection that isn't tied to a
    registered folder). Only returns an explicit list once the user has
    actually excluded a folder or put one into single-folder ("solo") mode,
    since at that point the default "search everything" is no longer correct.
    """
    folders = registry.list_folders()
    if not folders:
        return None

    solo: list[str] = []
    included: list[str] = []
    any_excluded = False

    for row in folders:
        cfg = registry.get_folder(row["id"])
        if cfg is None:
            continue
        meta = cfg.metadata or {}
        name = collection_for_folder(cfg.path)
        if meta.get("single_search"):
            solo.append(name)
        if meta.get("excluded"):
            any_excluded = True
        else:
            included.append(name)

    if solo:
        return solo
    if any_excluded:
        return included
    return None  # nothing toggled — search everything, including untracked collections


def run_search(
    query: str,
    top_k: int = 10,
    rewrite: bool = False,
    rerank: bool = False,
) -> tuple[list[SearchResult], str | None]:
    """Run the retrieval pipeline: optional rewrite -> multi-collection
    vector search -> merge -> optional rerank -> top_k.

    This mirrors ``mcp_server.tools.SragTools.search`` (the one proven
    end-to-end path in the codebase) rather than ``search.pipeline.SearchPipeline``,
    whose reranker/chroma-client wiring doesn't match the classes that
    actually exist (see investigation notes). Returns ``(results, rewritten_query)``.
    """
    store = get_store()
    registry = get_registry()
    embedder = get_embedder()
    collection_names = searchable_collection_names(registry)

    queries_to_embed = [query]
    rewritten_query: str | None = None
    if rewrite:
        rewriter = get_rewriter()
        if rewriter is not None:
            try:
                variants = rewriter.rewrite(query)
                extra = [v for v in variants if v != query]
                if extra:
                    rewritten_query = extra[0]
                    queries_to_embed.extend(extra)
            except Exception as exc:
                logger.warning("Query rewrite failed, using original query: %s", exc)

    raw_results: list[SearchResult] = []
    for q in queries_to_embed:
        try:
            query_embedding = embedder.embed(q)
        except Exception as exc:
            logger.error("Embedding query failed: %s", exc)
            continue
        raw_results.extend(
            store.multi_collection_search(
                query_embedding=query_embedding,
                top_k=top_k * 2,
                collection_names=collection_names,
            )
        )

    merged = ChromaStore.merge_results(raw_results, top_k=top_k * 2)

    if rerank and merged:
        reranker = get_reranker()
        if reranker is not None:
            try:
                merged = reranker.rerank(query, merged, top_k=top_k * 2)
            except Exception as exc:
                logger.warning("Reranking failed, using original order: %s", exc)

    return merged[:top_k], rewritten_query


# ─── Lifespan ──────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[dict]:
    """Startup/shutdown sequencing for all services.

    Startup  : config -> embedder -> store -> registry -> converter -> ingest queue worker
    Shutdown : stop ingest worker, close registry
    """
    logger.info("[startup] SRAG initializing...")
    get_config()
    get_embedder()
    get_store()
    get_registry()
    get_converter()

    from web.ingest_queue import get_ingest_queue
    get_ingest_queue().start()

    yield  # ── server runs here ───────────────────────────────────────

    logger.info("[shutdown] SRAG shutting down...")
    get_ingest_queue().stop()

    global _registry
    if _registry is not None:
        _registry.close()
        _registry = None

    logger.info("[shutdown] Done.")


__all__ = [
    "get_config",
    "get_embedder",
    "get_store",
    "get_registry",
    "get_converter",
    "get_reranker",
    "get_rewriter",
    "get_or_register_folder",
    "collection_for_folder",
    "searchable_collection_names",
    "run_search",
    "lifespan",
]
