"""SRAG Ingest Sync — Phase 2 implementation (DD-09).

Manages synchronization between the vector store, folder registry, and ingestion pipeline.
Supports full sync, incremental sync, and smart sync modes with automatic
change detection using file hashes and metadata from SQLite-based registry.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

from ingest.folder_registry import FileEntry, FolderConfig, FolderRegistry, ScanMode, collection_name_for_path
from ingest.pipeline import IngestCancelled, IngestResult, run as run_pipeline
from ingest.converter import DocumentConverter
from embedding import Embedder
from store.chromadb_store import ChromaStore

logger = logging.getLogger(__name__)


def _compute_file_hash(file_path: str) -> Optional[str]:
    """Compute SHA-256 hash of a file's content.

    Args:
        file_path: Path to the file.

    Returns:
        Hex digest or None if file cannot be read.
    """
    import hashlib
    try:
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                sha256.update(chunk)
        return sha256.hexdigest()
    except (OSError, IOError) as e:
        logger.warning("Failed to compute hash for '%s': %s", file_path, str(e))
        return None


def _detect_changes(
    folder_id: int,
    registry: FolderRegistry,
    excluded_patterns: Optional[list[str]] = None,
) -> tuple[list[FileEntry], list[dict[str, Any]], list[str]]:
    """Detect new and modified files for a monitored folder.

    Args:
        folder_id: ID of the folder in the registry.
        registry: FolderRegistry instance for change detection.
        excluded_patterns: Glob patterns to exclude from scanning.

    Returns:
        Tuple of (new_files, changed_files, unchanged_files) as lists of dicts
        with keys: path, file_size_bytes, last_modified.
    """
    from ingest.scanner import FileScanner

    folder = registry.get_folder(folder_id)
    if folder is None:
        logger.warning("Folder %d not found in registry", folder_id)
        return [], [], []

    # Scan actual files on disk
    scanner = FileScanner(excluded_patterns=excluded_patterns or [])
    scan_result = scanner.scan(str(Path(folder.path).resolve()))

    if scan_result is None:
        logger.warning("No files found in folder %d", folder_id)
        return [], [], []

    # Get current file entries from registry
    db_files = registry.list_files(folder_id)
    db_hash_map = {e.relative_path: e for e in db_files}

    new_files: list[FileEntry] = []
    changed_files: list[FileEntry] = []
    unchanged_files: list[str] = []

    # Check each scanned file
    for scan_file in scan_result.files:
        abs_path = str(scan_file.path)
        rel_path = str(Path(abs_path).relative_to(folder.path))

        current_hash = _compute_file_hash(abs_path)
        current_size = scan_file.size_bytes if hasattr(scan_file, 'size_bytes') else 0
        current_modified = scan_file.mtime if hasattr(scan_file, 'mtime') else time.time()

        # Check against registry entry
        db_entry = db_hash_map.get(rel_path)
        if db_entry is None:
            # New file not in registry
            new_files.append(FileEntry(
                folder_id=folder_id,
                relative_path=rel_path,
                file_hash=current_hash,
                file_size_bytes=current_size,
                last_modified=current_modified,
                status="new",
            ))
        elif db_entry.file_hash != current_hash:
            # Modified file (hash changed)
            changed_files.append(FileEntry(
                folder_id=folder_id,
                relative_path=rel_path,
                file_hash=current_hash,
                file_size_bytes=current_size,
                last_modified=current_modified,
                status="new",  # Will be reprocessed
            ))
        else:
            # Unchanged file (same hash)
            unchanged_files.append(rel_path)

    return new_files, changed_files, unchanged_files


def full_sync(
    folder_id: int,
    registry: FolderRegistry,
    converter: DocumentConverter,
    embedder: Embedder,
    store: ChromaStore,
    excluded_patterns: Optional[list[str]] = None,
) -> IngestResult:
    """Perform a full sync for a monitored folder.

    This processes ALL files in the folder regardless of previous state.
    Used for initial ingestion or complete re-ingestion.

    Args:
        folder_id: ID of the folder to sync.
        registry: FolderRegistry instance.
        converter: DocumentConverter for file conversion.
        embedder: Embedder for text embedding.
        store: ChromaStore to upsert chunks into and clean up deleted files from.
        excluded_patterns: Glob patterns to exclude.

    Returns:
        IngestResult with processing statistics.
    """
    folder = registry.get_folder(folder_id)
    if folder is None:
        raise ValueError(f"Folder {folder_id} not found in registry")

    logger.info("Starting full sync for folder '%s' (id=%d)", folder.path, folder_id)

    collection_name = collection_name_for_path(folder.path)

    # Run the ingestion pipeline (scan → convert → chunk → enrich → embed → upsert)
    result = run_pipeline(
        folder_path=folder.path,
        converter=converter,
        embedder=embedder,
        store=store,
        collection_name=collection_name,
        excluded_patterns=excluded_patterns,
    )

    if result.cancelled:
        logger.info("Full sync cancelled for folder '%s'", folder.path)
        return result

    # Update registry with processed files; remove chunks/entries for files
    # that no longer exist on disk.
    _update_registry_after_ingest(folder_id, registry, converter, folder.recursive, store, collection_name)

    logger.info("Full sync complete for folder '%s': %d files, %d chunks",
               folder.path, result.files_processed, result.chunks_created)

    return result


def incremental_sync(
    folder_id: int,
    registry: FolderRegistry,
    converter: DocumentConverter,
    embedder: Embedder,
    store: ChromaStore,
    excluded_patterns: Optional[list[str]] = None,
) -> IngestResult:
    """Perform an incremental sync for a monitored folder.

    Uses file hash comparison to detect new/modified files up front so a
    no-change run is a true no-op (skips the pipeline call entirely). The
    actual ingest pass still re-scans the whole folder — chunk IDs are
    deterministic, so re-upserting unchanged files is a no-op write, not a
    correctness issue — but note this means it isn't a *partial* reprocess.

    Args:
        folder_id: ID of the folder to sync.
        registry: FolderRegistry instance.
        converter: DocumentConverter for file conversion.
        embedder: Embedder for text embedding.
        store: ChromaStore to upsert chunks into and clean up deleted files from.
        excluded_patterns: Glob patterns to exclude.

    Returns:
        IngestResult with processing statistics.
    """
    folder = registry.get_folder(folder_id)
    if folder is None:
        raise ValueError(f"Folder {folder_id} not found in registry")

    logger.info("Starting incremental sync for folder '%s' (id=%d)", folder.path, folder_id)

    # Detect changes using file hashes
    new_files, changed_files, unchanged_files = _detect_changes(
        folder_id, registry, excluded_patterns
    )

    # _detect_changes only reports new/modified files present in the current
    # scan - a deletion-only sync (nothing new or modified, but a previously
    # registered file is now gone) would otherwise never reach the cleanup
    # pass below.
    still_present = set(unchanged_files) | {f.relative_path for f in new_files} | {f.relative_path for f in changed_files}
    registered_paths = {e.relative_path for e in registry.list_files(folder_id)}
    has_deletions = bool(registered_paths - still_present)

    if not new_files and not changed_files and not has_deletions:
        logger.info("No changes detected for folder '%s'", folder.path)
        return IngestResult(folder_path=folder.path, files_processed=0, chunks_created=0)

    logger.info("Changes detected: %d new, %d changed, %d unchanged",
               len(new_files), len(changed_files), len(unchanged_files))

    collection_name = collection_name_for_path(folder.path)

    # Run the ingestion pipeline (re-scans the folder; deterministic chunk
    # IDs make re-upserting unchanged files a no-op)
    result = run_pipeline(
        folder_path=folder.path,
        converter=converter,
        embedder=embedder,
        store=store,
        collection_name=collection_name,
        excluded_patterns=excluded_patterns,
    )

    if result.cancelled:
        logger.info("Incremental sync cancelled for folder '%s'", folder.path)
        return result

    # Update registry with new file entries and hashes; remove chunks/entries
    # for files that no longer exist on disk.
    _update_registry_after_ingest(folder_id, registry, converter, folder.recursive, store, collection_name)

    logger.info("Incremental sync complete: %d files processed, %d chunks created",
               result.files_processed, result.chunks_created)

    return result


def smart_sync(
    folder_id: int,
    registry: FolderRegistry,
    converter: DocumentConverter,
    embedder: Embedder,
    store: ChromaStore,
    excluded_patterns: Optional[list[str]] = None,
) -> IngestResult:
    """Perform a smart sync for a monitored folder.

    Combines new file detection with hash-based change detection.
    Uses directory timestamps for initial filtering before detailed hash checks.

    Args:
        folder_id: ID of the folder to sync.
        registry: FolderRegistry instance.
        converter: DocumentConverter for file conversion.
        embedder: Embedder for text embedding.
        store: ChromaStore to upsert chunks into and clean up deleted files from.
        excluded_patterns: Glob patterns to exclude.

    Returns:
        IngestResult with processing statistics.
    """
    folder = registry.get_folder(folder_id)
    if folder is None:
        raise ValueError(f"Folder {folder_id} not found in registry")

    logger.info("Starting smart sync for folder '%s' (id=%d)", folder.path, folder_id)

    # Use hash-based change detection (smart sync = incremental with detailed analysis)
    result = incremental_sync(
        folder_id=folder_id,
        registry=registry,
        converter=converter,
        embedder=embedder,
        store=store,
        excluded_patterns=excluded_patterns,
    )

    logger.info("Smart sync complete for folder '%s': %d files, %d chunks",
               folder.path, result.files_processed, result.chunks_created)

    return result


def _update_registry_after_ingest(
    folder_id: int,
    registry: FolderRegistry,
    converter: DocumentConverter,
    recursive: bool = True,
    store: ChromaStore | None = None,
    collection_name: str | None = None,
) -> None:
    """Update the folder registry after successful ingestion.

    Updates file hashes, sizes, modification times, and status for all
    files in the ingested folder. Files no longer on disk have their chunks
    removed from *store* (when provided) and their registry entry deleted.

    Args:
        folder_id: ID of the synced folder.
        registry: FolderRegistry instance.
        converter: DocumentConverter (for file type detection).
        recursive: Whether to include subdirectories.
        store: ChromaStore to remove deleted files' chunks from. If ``None``,
            deleted files are only dropped from the registry (matches prior
            behavior for callers that don't have a store handy).
        collection_name: Collection to delete stale chunks from; required if
            *store* is given.
    """
    from ingest.scanner import FileScanner

    folder = registry.get_folder(folder_id)
    if folder is None:
        return

    # Scan current files on disk
    scanner = FileScanner(excluded_patterns=folder.excluded_patterns or [])
    scan_result = scanner.scan(str(Path(folder.path).resolve()), recursive=recursive)

    if scan_result is None:
        return

    # Get existing registry entries
    db_files = registry.list_files(folder_id)
    db_path_set = {e.relative_path for e in db_files}

    # Build set of currently existing file paths
    current_paths: set[str] = set()

    for scan_file in scan_result.files:
        abs_path = str(scan_file.path)
        rel_path = str(Path(abs_path).relative_to(folder.path))
        current_paths.add(rel_path)

        # Compute hash and update registry entry
        file_hash = _compute_file_hash(abs_path)

        existing_entry = None
        for e in db_files:
            if e.relative_path == rel_path:
                existing_entry = e
                break

        if existing_entry:
            # Update existing entry with new hash/metadata
            registry.update_file(
                folder_id,
                rel_path,
                file_hash=file_hash,
                file_size_bytes=scan_file.size_bytes if hasattr(scan_file, 'size_bytes') else 0,
                last_modified=scan_file.mtime if hasattr(scan_file, 'mtime') else time.time(),
            )
        else:
            # Register new entry in registry
            from ingest.folder_registry import FileEntry
            new_entry = FileEntry(
                folder_id=folder_id,
                relative_path=rel_path,
                file_hash=file_hash,
                file_size_bytes=scan_file.size_bytes if hasattr(scan_file, 'size_bytes') else 0,
                last_modified=scan_file.mtime if hasattr(scan_file, 'mtime') else time.time(),
                status="processed",
            )
            registry.add_file(new_entry)

    # Remove files that are in the registry but no longer on disk: drop
    # their chunks from the collection (if a store was given) and delete
    # their registry entry, so a deleted file's content stops being
    # returned by search instead of lingering forever.
    for db_path in db_path_set:
        if db_path in current_paths:
            continue

        if store is not None and collection_name is not None:
            abs_path = str(Path(folder.path).resolve() / db_path)
            try:
                col = store.get_or_create_collection(collection_name)
                matches = col.get(where={"source": abs_path})
                stale_ids = matches.get("ids") or []
                if stale_ids:
                    store.delete(collection_name, ids=stale_ids)
                    logger.info(
                        "Removed %d stale chunk(s) for deleted file '%s' from '%s'.",
                        len(stale_ids), db_path, collection_name,
                    )
            except Exception as exc:
                logger.warning("Failed to remove chunks for deleted file '%s': %s", db_path, exc)

        registry.remove_file(folder_id, db_path)
        logger.info("Removed deleted file from registry: %s", db_path)


__all__ = [
    "full_sync",
    "incremental_sync",
    "smart_sync",
]