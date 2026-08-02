"""SRAG Folder Registry — Phase 2 implementation (DD-06).

Manages persistent tracking of monitored folders, file metadata, and change
detection for incremental ingestion. Uses SQLite as the backend store.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def collection_name_for_path(folder_path: str) -> str:
    """Deterministic ChromaDB collection name for a registered folder (DD-05).

    Bare sha256 of the normalized absolute path, no prefix — stable across
    path spellings (case, slash direction) so the same folder always maps to
    the same collection.
    """
    normalized = os.path.normcase(os.path.abspath(folder_path))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ScanMode(str, Enum):
    """Folder scanning mode."""

    FULL = "full"  # Always scan entire folder
    INCREMENTAL = "incremental"  # Only process changed files
    SMART = "smart"  # Detect new + modified files


class FolderState(str, Enum):
    """Folder monitoring state."""

    ACTIVE = "active"
    PAUSED = "paused"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class FolderConfig:
    """Configuration for a monitored folder.

    Attributes
    ----------
    path : str
        Absolute or relative path to the folder.
    scan_mode : ScanMode
        How to scan this folder.
    recursive : bool
        Whether to scan subdirectories.
    excluded_patterns : list[str]
        Glob patterns to exclude (e.g., ``["*.tmp", "__pycache__/*"]``).
    metadata : dict[str, Any]
        Additional user-defined metadata.
    polling_interval_sec : float
        Interval between scans in *incremental* mode.
    """

    path: str
    scan_mode: ScanMode = ScanMode.INCREMENTAL
    recursive: bool = True
    excluded_patterns: list[str] = None  # type: ignore[assignment]
    metadata: dict[str, Any] = None  # type: ignore[assignment]
    polling_interval_sec: float = 60.0

    def __post_init__(self) -> None:
        if self.excluded_patterns is None:
            self.excluded_patterns = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class FileEntry:
    """Record for a single tracked file.

    Attributes
    ----------
    folder_id : int
        Foreign key to the parent folder.
    relative_path : str
        Path relative to the folder root.
    file_hash : str | None
        Content hash (None = not yet computed).
    file_size_bytes : int
        File size in bytes.
    last_modified : float
        Last modification timestamp.
    status : str
        One of ``"new"``, ``"processed"``, ``"error"``.
    metadata : dict[str, Any]
        Additional per-file metadata.
    """

    folder_id: int
    relative_path: str
    file_hash: str | None = None
    file_size_bytes: int = 0
    last_modified: float = 0.0
    status: str = "new"
    metadata: dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


# ---------------------------------------------------------------------------
# FolderRegistry
# ---------------------------------------------------------------------------

class FolderRegistry:
    """SQLite-backed registry for monitored folders and their files.

    Example
    -------
    >>> reg = FolderRegistry("/tmp/srag_registry.db")
    >>> fid = reg.add_folder(FolderConfig(path="/data/docs"))
    >>> entries = reg.list_files(fid)
    >>> reg.update_file_status(fid, "sub/doc.pdf", status="processed")
    """

    # SQL schema version.
    SCHEMA_VERSION = 1

    # SQL statements.
    CREATE_FOLDERS_TABLE = """
        CREATE TABLE IF NOT EXISTS folders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT UNIQUE NOT NULL,
            scan_mode TEXT NOT NULL DEFAULT 'incremental',
            recursive INTEGER NOT NULL DEFAULT 1,
            excluded_patterns TEXT,
            metadata TEXT,
            state TEXT NOT NULL DEFAULT 'active',
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        )
    """

    CREATE_FILES_TABLE = """
        CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            folder_id INTEGER NOT NULL,
            relative_path TEXT NOT NULL,
            file_hash TEXT,
            file_size_bytes INTEGER NOT NULL DEFAULT 0,
            last_modified REAL NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'new',
            metadata TEXT,
            updated_at REAL NOT NULL DEFAULT 0,
            UNIQUE(folder_id, relative_path),
            FOREIGN KEY (folder_id) REFERENCES folders(id) ON DELETE CASCADE
        )
    """

    CREATE_INDEXES = [
        "CREATE INDEX IF NOT EXISTS idx_files_folder ON files(folder_id)",
        "CREATE INDEX IF NOT EXISTS idx_files_status ON files(status)",
        "CREATE INDEX IF NOT EXISTS idx_files_modified ON files(last_modified)",
    ]

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        """Initialize the registry with an SQLite database.

        Parameters
        ----------
        db_path : str | Path
            Path to the SQLite database file. Use ``":memory:"`` for ephemeral storage.
        """
        self._db_path = Path(db_path) if db_path != ":memory:" else None
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Return or create the SQLite connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self._db_path) if self._db_path else ":memory:",
                check_same_thread=False,
            )
            self._conn.row_factory = sqlite3.Row
            # Enable WAL mode for better concurrency.
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    def _init_db(self) -> None:
        """Create tables and indexes if they don't exist."""
        conn = self._get_connection()
        conn.executescript(self.CREATE_FOLDERS_TABLE)
        conn.executescript(self.CREATE_FILES_TABLE)
        for stmt in self.CREATE_INDEXES:
            conn.execute(stmt)
        conn.commit()
        logger.info("FolderRegistry initialized at %s", self._db_path or ":memory:")

    # ------------------------------------------------------------------
    # Folder CRUD
    # ------------------------------------------------------------------

    def add_folder(self, config: FolderConfig) -> int:
        """Register a new monitored folder.

        Parameters
        ----------
        config : FolderConfig
            Configuration for the folder.

        Returns
        -------
        int
            The assigned folder ID.

        Raises
        ------
        ValueError
            If the folder is already registered.
        """
        conn = self._get_connection()
        now = time.time()
        try:
            cursor = conn.execute(
                """INSERT INTO folders (path, scan_mode, recursive, excluded_patterns, metadata, state, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(Path(config.path).resolve()),
                    config.scan_mode.value,
                    int(config.recursive),
                    json.dumps(config.excluded_patterns),
                    json.dumps(config.metadata),
                    FolderState.ACTIVE.value,
                    now,
                    now,
                ),
            )
            conn.commit()
            folder_id = cursor.lastrowid
            logger.info("Added folder '%s' (id=%d).", config.path, folder_id)
            return folder_id
        except sqlite3.IntegrityError:
            raise ValueError(f"Folder already registered: {config.path}")

    def get_folder(self, folder_id: int) -> FolderConfig | None:
        """Retrieve configuration for *folder_id*."""
        conn = self._get_connection()
        row = conn.execute("SELECT * FROM folders WHERE id = ?", (folder_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_folder_config(row)

    def list_folders(self, state: FolderState | None = None) -> list[dict[str, Any]]:
        """List all registered folders.

        Parameters
        ----------
        state : FolderState | None
            Filter by state. ``None`` returns all.

        Returns
        -------
        list[dict]
            Each dict has keys: ``id``, ``path``, ``scan_mode``, ``recursive``,
            ``state``, ``created_at``, ``updated_at``.
        """
        conn = self._get_connection()
        query = "SELECT id, path, scan_mode, recursive, state, created_at, updated_at FROM folders"
        params: list[Any] = []
        if state is not None:
            query += " WHERE state = ?"
            params.append(state.value)

        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def update_folder_state(self, folder_id: int, state: FolderState) -> bool:
        """Update the monitoring *state* of a folder.

        Returns ``True`` on success.
        """
        conn = self._get_connection()
        now = time.time()
        cursor = conn.execute(
            "UPDATE folders SET state = ?, updated_at = ? WHERE id = ?",
            (state.value, now, folder_id),
        )
        conn.commit()
        return cursor.rowcount > 0

    def remove_folder(self, folder_id: int) -> bool:
        """Unregister a folder and all its file entries."""
        conn = self._get_connection()
        cursor = conn.execute("DELETE FROM folders WHERE id = ?", (folder_id,))
        conn.commit()
        if cursor.rowcount == 0:
            return False
        logger.info("Removed folder id=%d.", folder_id)
        return True

    def get_folder_by_path(self, path: str) -> tuple[int, FolderConfig] | None:
        """Look up a folder by its resolved absolute path.

        Returns ``(folder_id, FolderConfig)`` or ``None`` if not registered.
        """
        conn = self._get_connection()
        resolved = str(Path(path).resolve())
        row = conn.execute("SELECT * FROM folders WHERE path = ?", (resolved,)).fetchone()
        if row is None:
            return None
        return row["id"], self._row_to_folder_config(row)

    def update_folder_metadata(self, folder_id: int, **updates: Any) -> bool:
        """Merge *updates* into a folder's ``metadata`` JSON column.

        This is the web layer's storage for fields the DD-06 schema doesn't
        have dedicated columns for (ingest status, cached doc/chunk counts,
        excluded/single-search flags, last error) — additive, so it needs no
        schema migration.
        """
        conn = self._get_connection()
        row = conn.execute("SELECT metadata FROM folders WHERE id = ?", (folder_id,)).fetchone()
        if row is None:
            return False
        current = json.loads(row["metadata"]) if row["metadata"] else {}
        current.update(updates)
        conn.execute(
            "UPDATE folders SET metadata = ?, updated_at = ? WHERE id = ?",
            (json.dumps(current), time.time(), folder_id),
        )
        conn.commit()
        return True

    # ------------------------------------------------------------------
    # File CRUD
    # ------------------------------------------------------------------

    def add_file(self, entry: FileEntry) -> int:
        """Register a file in the registry.

        Returns the assigned file ID or -1 on conflict.
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """INSERT OR IGNORE INTO files (folder_id, relative_path, file_hash, file_size_bytes, last_modified, status, metadata, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    entry.folder_id,
                    entry.relative_path,
                    entry.file_hash,
                    entry.file_size_bytes,
                    entry.last_modified,
                    entry.status,
                    json.dumps(entry.metadata),
                    time.time(),
                ),
            )
            conn.commit()
            return cursor.lastrowid or -1
        except sqlite3.IntegrityError:
            return -1

    def update_file(self, folder_id: int, relative_path: str, **kwargs: Any) -> bool:
        """Update file metadata for *(folder_id, relative_path)*.

        Parameters
        ----------
        folder_id : int
        relative_path : str
        **kwargs : Any
            Keys can include ``file_hash``, ``file_size_bytes``, ``last_modified``,
            ``status``, ``metadata``.

        Returns
        -------
        bool
        """
        conn = self._get_connection()
        updates: list[tuple[str, Any]] = []
        values: list[Any] = []

        for key, val in kwargs.items():
            if key == "status":
                updates.append(("status", val))
            elif key == "metadata":
                updates.append(("metadata", json.dumps(val)))
            else:
                updates.append((key, val))

        if not updates:
            return False

        set_clause = ", ".join(f"{k} = ?" for k, _ in updates)
        values = [v for _, v in updates] + [time.time(), relative_path, folder_id]

        conn.execute(
            f"UPDATE files SET {set_clause}, updated_at = ? WHERE folder_id = ? AND relative_path = ?",
            values,
        )
        conn.commit()
        return True

    def update_file_status(self, folder_id: int, relative_path: str, status: str) -> bool:
        """Convenience method to update only the *status* field."""
        return self.update_file(folder_id, relative_path, status=status)

    def get_file(self, folder_id: int, relative_path: str) -> FileEntry | None:
        """Retrieve file entry for *(folder_id, relative_path)*."""
        conn = self._get_connection()
        row = conn.execute(
            "SELECT * FROM files WHERE folder_id = ? AND relative_path = ?",
            (folder_id, relative_path),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_file_entry(row)

    def list_files(self, folder_id: int, status: str | None = None) -> list[FileEntry]:
        """List files for *folder_id*, optionally filtered by *status*."""
        conn = self._get_connection()
        query = "SELECT * FROM files WHERE folder_id = ?"
        params: list[Any] = [folder_id]
        if status is not None:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY last_modified DESC"

        rows = conn.execute(query, params).fetchall()
        return [self._row_to_file_entry(r) for r in rows]

    def get_new_files(self, folder_id: int) -> list[FileEntry]:
        """Return files with ``status='new'``."""
        return self.list_files(folder_id, status="new")

    def mark_all_processed(self, folder_id: int) -> int:
        """Mark all files in *folder_id* as ``'processed'``. Returns count updated."""
        conn = self._get_connection()
        cursor = conn.execute(
            "UPDATE files SET status = 'processed', updated_at = ? WHERE status = 'new' AND folder_id = ?",
            (time.time(), folder_id),
        )
        conn.commit()
        return cursor.rowcount

    def remove_file(self, folder_id: int, relative_path: str) -> bool:
        """Delete a file entry from the registry."""
        conn = self._get_connection()
        cursor = conn.execute(
            "DELETE FROM files WHERE folder_id = ? AND relative_path = ?",
            (folder_id, relative_path),
        )
        conn.commit()
        return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _row_to_folder_config(self, row: sqlite3.Row) -> FolderConfig:
        """Convert a SQLite row to :py:class:`FolderConfig`."""
        return FolderConfig(
            path=row["path"],
            scan_mode=ScanMode(row["scan_mode"]),
            recursive=bool(row["recursive"]),
            excluded_patterns=json.loads(row["excluded_patterns"]) if row["excluded_patterns"] else [],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )

    def _row_to_file_entry(self, row: sqlite3.Row) -> FileEntry:
        """Convert a SQLite row to :py:class:`FileEntry`."""
        return FileEntry(
            folder_id=row["folder_id"],
            relative_path=row["relative_path"],
            file_hash=row["file_hash"],
            file_size_bytes=row["file_size_bytes"],
            last_modified=row["last_modified"] or 0.0,
            status=row["status"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            logger.info("FolderRegistry closed.")

    def __enter__(self) -> FolderRegistry:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


__all__ = [
    "ScanMode",
    "FolderState",
    "FolderConfig",
    "FileEntry",
    "FolderRegistry",
    "collection_name_for_path",
]