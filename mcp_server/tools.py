"""SRAG MCP Tools (FD-82).

Defines all MCP tools and resources exposed by the SRAG MCP server.
Each tool wraps an existing service-layer function from other modules.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from core.config import load_config
from core.embedding import get_embedder
from llm.factory import create_client as get_llm_client
from store.chromadb_store import SearchResult, ChromaStore

# ──────────────────── Data Classes ─────────────────────


@dataclass
class SearchResultDC:
    """Serialized search result."""
    id: str
    title: str
    content: str
    score: float
    source_file: str
    chunk_index: int | None = None

    @classmethod
    def from_store_result(cls, r: SearchResult) -> "SearchResultDC":
        return cls(
            id=r.text if r.text else str(uuid.uuid4()),
            title=r.metadata.get("title", ""),
            content=r.metadata.get("content", r.text),
            score=float(r.distance),
            source_file=r.metadata.get("source_file", r.metadata.get("filename", "")),
            chunk_index=r.metadata.get("chunk_index"),
        )


# ──────────────────── Tool Definitions ─────────────────

class SragTools:
    """Collection of MCP tools for SRAG."""

    # -- Search tools ----------------------------------

    @staticmethod
    def search(
        query: str,
        top_k: int = 8,
        min_score: float = 0.15,
        rewrite_query: bool = False,
        rerank_results: bool = False,
        category: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search the knowledge base for relevant content."""
        cfg = load_config()
        store = ChromaStore(persist_directory=cfg.chromadb_path)

        # Step 1 - Rewrite query (FD-57).
        queries_to_embed = [query]
        if rewrite_query:
            try:
                from search.rewriter import QueryRewriter
                variants = QueryRewriter().rewrite(query)
                queries_to_embed.extend(variants)
            except Exception:
                pass

        # Step 2 - Embed + retrieve.
        embedder = get_embedder()
        raw_results: list[SearchResult] = []
        for q in queries_to_embed:
            try:
                query_embedding = embedder.embed(q)
            except Exception:
                continue
            all_results = store.multi_collection_search(
                query_embedding=query_embedding,
                top_k=top_k * 2,
                collection_names=None,  # search every collection that exists
            )
            raw_results.extend(r for r in all_results if r.distance <= (1.0 - min_score))

        # Step 3 - Merge (using store's merge_results).
        merged: list[SearchResult] = ChromaStore.merge_results(raw_results, top_k=top_k * 2)

        # Step 4 - Rerank (FD-58).
        if rerank_results and merged:
            try:
                from search.reranker import Reranker
                merged = Reranker().rerank(query, merged, top_k=top_k * 2)
            except Exception:
                pass

        # Step 5 - Trim & serialize.
        final = merged[:top_k]
        return [asdict(SearchResultDC.from_store_result(r)) for r in final]

    # -- Ingest tools ----------------------------------
    #
    # The ingest pipeline operates per-folder (FolderRegistry + IngestQueue),
    # not per-file, so these tools resolve a target path to its owning
    # registered folder (registering it on first use) before queueing.

    @staticmethod
    def ingest_file(file_path: str) -> dict[str, Any]:
        """Ingest a single file into the knowledge base.

        The containing folder is registered/ingested as a whole — the
        pipeline re-scans and upserts idempotently, so this is safe even if
        the folder has been ingested before.
        """
        fpath = Path(file_path)
        if not fpath.exists():
            return {"status": "error", "error": f"File not found: {file_path}"}

        try:
            from web.dependencies import get_or_register_folder
            from web.ingest_queue import get_ingest_queue
            folder_id, folder_path = get_or_register_folder(str(fpath.resolve()))
            task_id = get_ingest_queue().push(folder_id, folder_path)
            return {
                "status": "queued",
                "task_id": task_id,
                "folder": folder_path,
            }
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    @staticmethod
    def ingest_folder(folder_path: str, recursive: bool = True) -> dict[str, Any]:
        """Ingest all supported files from a folder."""
        fpath = Path(folder_path)
        if not fpath.is_dir():
            return {"status": "error", "error": f"Not a directory: {folder_path}"}

        try:
            from web.dependencies import get_or_register_folder
            from web.ingest_queue import get_ingest_queue
            folder_id, resolved_path = get_or_register_folder(str(fpath.resolve()))
            task_id = get_ingest_queue().push(folder_id, resolved_path)
            return {
                "status": "queued",
                "task_id": task_id,
                "folder": resolved_path,
            }
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    @staticmethod
    def folder_sync(folders: list[dict[str, Any]]) -> dict[str, Any]:
        """Queue a re-ingest for each given folder (FD-41), registering new ones.

        Args:
            folders: List of dicts with at least a ``path`` key (or bare
                path strings).
        """
        try:
            from web.dependencies import get_or_register_folder
            from web.ingest_queue import get_ingest_queue
            queue = get_ingest_queue()

            task_ids: list[str] = []
            for f in folders:
                path = f.get("path") if isinstance(f, dict) else f
                if not path:
                    continue
                folder_id, folder_path = get_or_register_folder(path)
                task_ids.append(queue.push(folder_id, folder_path, kind="sync"))

            return {"status": "queued", "synced": len(task_ids), "task_ids": task_ids}
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    # -- Knowledge management --------------------------

    @staticmethod
    def get_knowledge_stats() -> dict[str, Any]:
        """Get knowledge base statistics (FD-42)."""
        cfg = load_config()
        store = ChromaStore(persist_directory=cfg.chromadb_path)
        collections = store.list_collections()
        total_docs = sum(c.get("count", 0) for c in collections)
        return {
            "total_documents": total_docs,
            "total_chunks": total_docs,
            "by_category": {},
            "by_source_type": {},
            "collections": collections,
        }

    @staticmethod
    def list_knowledge_sources(limit: int = 50, offset: int = 0) -> dict[str, Any]:
        """List knowledge sources with pagination."""
        cfg = load_config()
        store = ChromaStore(persist_directory=cfg.chromadb_path)
        collections = store.list_collections()
        total = len(collections)
        return {
            "sources": collections[offset:offset + limit],
            "total": total,
        }

    @staticmethod
    def delete_knowledge_item(item_id: str) -> dict[str, Any]:
        """Delete a knowledge item by ID."""
        cfg = load_config()
        store = ChromaStore(persist_directory=cfg.chromadb_path)
        try:
            for name in [c["name"] for c in store.list_collections()]:
                col = store.get_or_create_collection(name)
                col.delete(ids=[item_id])
            return {"status": "completed", "deleted_id": item_id}
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    # -- LLM tools -------------------------------------

    @staticmethod
    def get_llm_config() -> dict[str, Any]:
        """Get current LLM configuration."""
        cfg = load_config()
        return {
            "provider": getattr(cfg, 'llm_provider', getattr(cfg, 'provider', 'default')),
            "model": getattr(cfg, 'embedding_model', getattr(cfg, 'model', 'default')),
            "temperature": 0.7,
            "max_tokens": 4096,
        }

    @staticmethod
    def set_llm_config(provider: str | None = None, model: str | None = None) -> dict[str, Any]:
        """Set LLM configuration dynamically (runtime-only, not persisted)."""
        cfg = load_config()
        if provider:
            cfg.llm_provider = provider
        if model:
            cfg.embedding_model = model
        return {
            "provider": getattr(cfg, 'llm_provider', ''),
            "model": getattr(cfg, 'embedding_model', ''),
        }

    # -- System tools ----------------------------------

    @staticmethod
    def health_check() -> dict[str, Any]:
        """Check system health and connectivity."""
        result = {"status": "healthy", "components": {}}

        # Storage check.
        try:
            cfg = load_config()
            store = ChromaStore(persist_directory=cfg.chromadb_path)
            store.initialize()
            collections = store.list_collections()
            total = sum(c.get("count", 0) for c in collections)
            result["components"]["storage"] = {"status": "ok", "collections": len(collections), "documents": total}
        except Exception as exc:
            result["components"]["storage"] = {"status": "error", "error": str(exc)}
            result["status"] = "degraded"

        # Embedding check.
        try:
            embedder = get_embedder()
            dim = embedder.dimension if hasattr(embedder, 'dimension') else 0
            result["components"]["embedding"] = {"status": "ok", "dimensions": dim}
        except Exception as exc:
            result["components"]["embedding"] = {"status": "error", "error": str(exc)}
            result["status"] = "degraded"

        # LLM check.
        try:
            client = get_llm_client()
            model_name = client.model_name if hasattr(client, 'model_name') else "unknown"
            result["components"]["llm"] = {"status": "ok", "model": model_name}
        except Exception as exc:
            result["components"]["llm"] = {"status": "error", "error": str(exc)}
            result["status"] = "degraded"

        return result

    @staticmethod
    def get_all_tools() -> list[dict[str, Any]]:
        """Return metadata for all available tools."""
        return [
            {"name": "search", "description": "Search the knowledge base"},
            {"name": "ingest_file", "description": "Ingest a single file"},
            {"name": "ingest_folder", "description": "Ingest all files in a folder"},
            {"name": "folder_sync", "description": "Synchronize folders for incremental updates"},
            {"name": "get_knowledge_stats", "description": "Get knowledge base statistics"},
            {"name": "list_knowledge_sources", "description": "List knowledge sources"},
            {"name": "delete_knowledge_item", "description": "Delete a knowledge item"},
            {"name": "get_llm_config", "description": "Get LLM configuration"},
            {"name": "set_llm_config", "description": "Set LLM configuration"},
            {"name": "health_check", "description": "Check system health"},
        ]

    @staticmethod
    def get_all_resources() -> list[dict[str, Any]]:
        """Return metadata for all available resources."""
        return [
            {"uri": "srag://config", "name": "System Configuration", "description": "Current SRAG configuration"},
            {"uri": "srag://stats", "name": "Knowledge Statistics", "description": "Knowledge base statistics"},
            {"uri": "srag://health", "name": "Health Status", "description": "System health status"},
        ]


# Singleton instance.
tools = SragTools()