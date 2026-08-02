"""SRAG HTTP Routes — REST API (SD.md §10).

16 endpoints under ``/api`` covering system status, folder registration and
ingestion, search, ChromaDB collections, and runtime configuration. The
``eval/*`` endpoints are intentionally omitted — PLAN.md marks the eval
package out of scope for this build.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from core.config import patch_config
from ingest.folder_registry import FolderConfig, FolderRegistry, collection_name_for_path
from ingest.scanner import scan_folder as scan_folder_files
from web.dependencies import (
    get_config,
    get_registry,
    get_store,
    run_search,
)
from web.ingest_queue import get_ingest_queue

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

# Runtime config keys the Settings page is allowed to PATCH. Anything else
# (paths, URLs, model names) is out of reach at runtime by design (FD-06).
_PATCHABLE_CONFIG_KEYS = {
    "search_rewrite_enabled",
    "search_rerank_enabled",
    "search_retrieval_k",
    "search_final_k",
}


# ─── Request / response models ────────────────────────────────────────

class FolderRegisterRequest(BaseModel):
    path: str = Field(..., min_length=1)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=10, ge=1, le=100)
    rewrite: bool = False
    rerank: bool = False


class ConfigPatchRequest(BaseModel):
    model_config = ConfigDict(extra="allow")


# ─── Helpers ───────────────────────────────────────────────────────────

def _folder_view(registry: FolderRegistry, folder_id: int, cfg: FolderConfig) -> dict[str, Any]:
    """Build the API-facing dict for one registered folder."""
    store = get_store()
    meta = cfg.metadata or {}
    collection_name = collection_name_for_path(cfg.path)
    chunk_count = store.collection_count(collection_name) if store.collection_exists(collection_name) else 0

    task = get_ingest_queue().get_status_for_folder(folder_id)
    status = meta.get("ingest_status", "registered")
    if task and task["status"] in ("running", "pending", "paused"):
        status = "indexing"

    return {
        "id": folder_id,
        "path": cfg.path,
        "name": Path(cfg.path).name,
        "status": status,
        "chunk_count": chunk_count,
        "excluded": bool(meta.get("excluded", False)),
        "single_search": bool(meta.get("single_search", False)),
        "last_indexed": meta.get("last_indexed"),
        "last_error": meta.get("last_error"),
        "task": task,
    }


def _resolve_folder(registry: FolderRegistry, folder_path: str) -> tuple[int, FolderConfig]:
    found = registry.get_folder_by_path(folder_path)
    if found is None:
        raise HTTPException(status_code=404, detail=f"Folder not registered: {folder_path}")
    return found


# ─── System status ─────────────────────────────────────────────────────

@router.get("/status")
async def api_status():
    """GET /api/status — folder count, total chunks, collection count, ChromaDB health."""
    registry = get_registry()
    store = get_store()
    folders = registry.list_folders()

    total_chunks = 0
    for row in folders:
        cfg = registry.get_folder(row["id"])
        if cfg is None:
            continue
        name = collection_name_for_path(cfg.path)
        if store.collection_exists(name):
            total_chunks += store.collection_count(name)

    try:
        collections = store.list_collections()
        chromadb_connected = True
    except Exception:
        collections = []
        chromadb_connected = False

    return {
        "folders": len(folders),
        "total_chunks": total_chunks,
        "collections": len(collections),
        "chromadb_connected": chromadb_connected,
    }


# ─── Folders ────────────────────────────────────────────────────────────

@router.get("/folders")
async def api_list_folders():
    registry = get_registry()
    folders = registry.list_folders()
    result = []
    for row in folders:
        cfg = registry.get_folder(row["id"])
        if cfg is not None:
            result.append(_folder_view(registry, row["id"], cfg))
    return {"folders": result}


@router.post("/folders", status_code=201)
async def api_register_folder(req: FolderRegisterRequest):
    path = req.path
    if not Path(path).is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {path}")

    registry = get_registry()
    try:
        folder_id = registry.add_folder(FolderConfig(path=path))
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    task_id = get_ingest_queue().push(folder_id, str(Path(path).resolve()), kind="ingest")
    cfg = registry.get_folder(folder_id)
    view = _folder_view(registry, folder_id, cfg)
    view["task_id"] = task_id
    return view


@router.delete("/folders/{folder_path:path}")
async def api_remove_folder(folder_path: str):
    registry = get_registry()
    folder_id, cfg = _resolve_folder(registry, folder_path)
    store = get_store()
    collection_name = collection_name_for_path(cfg.path)
    if store.collection_exists(collection_name):
        store.delete_collection(collection_name)
    registry.remove_folder(folder_id)
    return {"removed": True, "path": cfg.path}


@router.post("/folders/{folder_path:path}/ingest")
async def api_ingest_folder(folder_path: str):
    registry = get_registry()
    folder_id, cfg = _resolve_folder(registry, folder_path)
    task_id = get_ingest_queue().push(folder_id, cfg.path, kind="ingest")
    return {"task_id": task_id}


@router.post("/folders/{folder_path:path}/prioritize")
async def api_prioritize_folder(folder_path: str):
    registry = get_registry()
    folder_id, cfg = _resolve_folder(registry, folder_path)
    task_id = get_ingest_queue().prioritize(folder_id, cfg.path)
    return {"task_id": task_id}


@router.post("/folders/{folder_path:path}/sync")
async def api_sync_folder(folder_path: str):
    """Queue an incremental sync.

    Simplification: this currently runs the same idempotent full re-ingest as
    ``/ingest`` (safe — chunk IDs are deterministic upserts) rather than a
    true changed-files-only diff. True incremental sync is a follow-up.
    """
    registry = get_registry()
    folder_id, cfg = _resolve_folder(registry, folder_path)
    task_id = get_ingest_queue().push(folder_id, cfg.path, kind="sync")
    return {"task_id": task_id}


@router.get("/folders/{folder_path:path}/status")
async def api_folder_status(folder_path: str):
    registry = get_registry()
    folder_id, cfg = _resolve_folder(registry, folder_path)
    return _folder_view(registry, folder_id, cfg)


@router.get("/folders/{folder_path:path}/errors")
async def api_folder_errors(folder_path: str):
    registry = get_registry()
    folder_id, cfg = _resolve_folder(registry, folder_path)
    errors = (cfg.metadata or {}).get("last_errors", [])
    return {"errors": errors}


@router.get("/folders/{folder_path:path}/files")
async def api_folder_files(folder_path: str):
    registry = get_registry()
    _folder_id, cfg = _resolve_folder(registry, folder_path)
    files = scan_folder_files(cfg.path)
    return {
        "files": [
            {
                "path": f["path"],
                "size": f["size"],
                "modified": f["modified"],
                "format": f["format"].value if f["format"] is not None else None,
            }
            for f in files
        ]
    }


@router.patch("/folders/{folder_path:path}/toggle-excluded")
async def api_toggle_excluded(folder_path: str):
    registry = get_registry()
    folder_id, cfg = _resolve_folder(registry, folder_path)
    new_value = not bool((cfg.metadata or {}).get("excluded", False))
    registry.update_folder_metadata(folder_id, excluded=new_value)
    return {"excluded": new_value}


@router.patch("/folders/{folder_path:path}/toggle-single")
async def api_toggle_single(folder_path: str):
    registry = get_registry()
    folder_id, cfg = _resolve_folder(registry, folder_path)
    new_value = not bool((cfg.metadata or {}).get("single_search", False))
    registry.update_folder_metadata(folder_id, single_search=new_value)
    return {"single_search": new_value}


# ─── Search ─────────────────────────────────────────────────────────────

@router.post("/search")
async def api_search(req: SearchRequest):
    try:
        results, rewritten_query = run_search(
            query=req.query, top_k=req.top_k, rewrite=req.rewrite, rerank=req.rerank,
        )
    except Exception as exc:
        logger.exception("Search failed for query: %s", req.query)
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}")

    registry = get_registry()
    label_by_collection: dict[str, str] = {}
    for row in registry.list_folders():
        fcfg = registry.get_folder(row["id"])
        if fcfg is not None:
            label_by_collection[collection_name_for_path(fcfg.path)] = Path(fcfg.path).name

    serialized = []
    for r in results:
        meta = r.metadata or {}
        serialized.append({
            "text": r.text,
            "filename": meta.get("filename"),
            "source": meta.get("source"),
            "chunk_index": meta.get("chunk_index"),
            "total_chunks": meta.get("total_chunks"),
            "distance": r.distance,
            "score": max(0.0, 1.0 - r.distance),
            "rerank_score": getattr(r, "rerank_score", None),
            "collection": label_by_collection.get(r.collection, r.collection),
        })

    return {
        "query": req.query,
        "rewritten_query": rewritten_query,
        "reranked": req.rerank,
        "results": serialized,
    }


# ─── Collections ────────────────────────────────────────────────────────

@router.get("/collections")
async def api_collections():
    store = get_store()
    registry = get_registry()

    path_by_collection: dict[str, str] = {}
    for row in registry.list_folders():
        cfg = registry.get_folder(row["id"])
        if cfg is not None:
            path_by_collection[collection_name_for_path(cfg.path)] = cfg.path

    collections = []
    for c in store.list_collections():
        folder_path = path_by_collection.get(c["name"])
        collections.append({
            "name": c["name"],
            "count": c["count"],
            "folder_path": folder_path,
            "label": Path(folder_path).name if folder_path else c["name"][:12],
        })
    return {"collections": collections}


# ─── Config ─────────────────────────────────────────────────────────────

@router.get("/config")
async def api_get_config():
    cfg = get_config()
    return {
        "chromadb_path": cfg.chromadb_path,
        "ollama_url": cfg.ollama_url,
        "embedding_model": cfg.embedding_model,
        "server_port": cfg.server_port,
        "chunking_max_tokens": cfg.chunking_max_tokens,
        "chunking_overlap": cfg.chunking_overlap,
        "search_rewrite_enabled": cfg.search_rewrite_enabled,
        "search_rerank_enabled": cfg.search_rerank_enabled,
        "search_retrieval_k": cfg.search_retrieval_k,
        "search_final_k": cfg.search_final_k,
        "llm_rewrite": cfg.llm_rewrite,
        "llm_rerank": cfg.llm_rerank,
        "llm_enrichment": cfg.llm_enrichment,
    }


@router.patch("/config")
async def api_patch_config(req: ConfigPatchRequest):
    updates = req.model_dump(exclude_unset=True)
    unknown = set(updates) - _PATCHABLE_CONFIG_KEYS
    if unknown:
        raise HTTPException(status_code=400, detail=f"Not runtime-patchable: {sorted(unknown)}")
    cfg = patch_config(get_config(), updates)
    return {"updated": updates, "search_rewrite_enabled": cfg.search_rewrite_enabled,
            "search_rerank_enabled": cfg.search_rerank_enabled}


# ─── Router registration ────────────────────────────────────────────────

def register_routes(app: Any) -> None:
    app.include_router(router, tags=["api"])
    logger.info("Web routes registered (%d endpoints).", len(router.routes))
