"""SRAG File Scanner — Phase 2 implementation (DD-07).

Scans filesystem directories to discover files matching supported document
formats. Compares discovered files against the FolderRegistry for incremental
updates.
"""

from __future__ import annotations

import fnmatch
import hashlib
import logging
import os
from pathlib import Path
from typing import Any

from dataclasses import dataclass, field

from ingest.converter import DocFormat
from ingest.folder_registry import FileEntry, FileEntry as _FE, FolderConfig, FolderRegistry

logger = logging.getLogger(__name__)

# Supported extensions (lowercase).
SUPPORTED_EXTENSIONS: set[str] = {
    ".pdf",
    ".docx",
    ".doc",
    ".txt",
    ".md",
    ".markdown",
    ".epub",
    ".html",
    ".htm",
    ".png",
    ".jpg",
    ".jpeg",
}

# DocFormat has no `from_extension` classmethod — this is the extension -> format
# mapping used by scan_folder() below (kept in sync with SUPPORTED_EXTENSIONS).
_EXT_TO_DOC_FORMAT: dict[str, DocFormat] = {
    ".pdf": DocFormat.PDF,
    ".docx": DocFormat.DOCX,
    ".doc": DocFormat.DOCX,
    ".txt": DocFormat.TXT,
    ".md": DocFormat.MD,
    ".markdown": DocFormat.MD,
    ".epub": DocFormat.EPUB,
    ".html": DocFormat.HTML,
    ".htm": DocFormat.HTML,
    ".png": DocFormat.IMAGE,
    ".jpg": DocFormat.IMAGE,
    ".jpeg": DocFormat.IMAGE,
}


@dataclass
class ScannedFile:
    """A single file discovered by a standalone (registry-less) scan."""

    path: Path
    size_bytes: int = 0
    mtime: float = 0.0
    hash: str | None = None


@dataclass
class ScanResult:
    """Result of :meth:`FileScanner.scan` — a standalone folder scan."""

    folder_path: str
    files: list[ScannedFile] = field(default_factory=list)


class FileScanner:
    """Discover files in monitored folders and sync with FolderRegistry.

    Example
    -------
    >>> reg = FolderRegistry()
    >>> scanner = FileScanner(reg)
    >>> new_files = scanner.scan_folder(folder_id=1)
    >>> for fpath, meta in new_files:
    ...     print(f"Found: {fpath}")
    """

    def __init__(
        self,
        registry: FolderRegistry | None = None,
        excluded_patterns: list[str] | None = None,
    ) -> None:
        """Initialize the scanner.

        Parameters
        ----------
        registry : FolderRegistry | None
            The shared folder registry instance for tracking file states.
            Required for registry-backed scanning (``scan_folder``/``scan_folders``).
        excluded_patterns : list[str] | None
            Glob patterns to exclude. Used by the standalone ``scan()`` method,
            which does not require a registry.
        """
        self._registry = registry
        self._excluded_patterns = excluded_patterns or []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan_folder(self, folder_id: int) -> list[tuple[Path, dict[str, Any]]]:
        """Scan a registered folder and return new/changed files.

        Parameters
        ----------
        folder_id : int
            The folder ID from the registry.

        Returns
        -------
        list[tuple[Path, dict]]
            Each tuple is ``(absolute_path, file_metadata)`` where metadata
            contains ``size``, ``modified``, ``hash``.
        """
        config = self._registry.get_folder(folder_id)
        if config is None:
            logger.warning("Folder id=%d not found in registry.", folder_id)
            return []

        if self._registry.get_folder(folder_id) is None:
            logger.warning("Folder id=%d does not exist; skipping scan.", folder_id)
            return []

        root = Path(config.path).resolve()
        if not root.is_dir():
            logger.error("Scan target '%s' is not a directory.", root)
            return []

        files = self._discover_files(root, config)
        changed = self._detect_changes(folder_id, root, files)

        logger.info(
            "Scanned folder '%s' (id=%d): %d file(s) found (%d new, %d modified).",
            config.path,
            folder_id,
            len(changed),
            sum(1 for _, s in changed if s == "new"),
            sum(1 for _, s in changed if s != "new"),
        )

        return [(p, m) for p, (_, m) in changed]

    def scan_folders(
        self,
        folder_ids: list[int] | None = None,
        state: str = "active",
    ) -> dict[int, list[tuple[Path, dict[str, Any]]]]:
        """Scan multiple folders.

        Parameters
        ----------
        folder_ids : list[int] | None
            Specific folder IDs to scan. ``None`` scans all active folders.
        state : str
            Filter folders by state. Defaults to ``"active"``.

        Returns
        -------
        dict[int, list[tuple[Path, dict]]]
            Maps folder_id → list of (path, metadata) tuples.
        """
        if folder_ids is not None:
            folders = [fid for fid in folder_ids if self._registry.get_folder(fid) is not None]
        else:
            folders = [
                f["id"]
                for f in self._registry.list_folders(state=state)
            ]

        results: dict[int, list[tuple[Path, dict[str, Any]]]] = {}
        for fid in folders:
            try:
                results[fid] = self.scan_folder(fid)
            except Exception as exc:
                logger.error("Failed to scan folder id=%d: %s", fid, exc)
                # Attempt to mark the folder as errored.
                try:
                    from ingest.folder_registry import FolderState  # noqa: F401
                except ImportError:
                    pass

        return results

    # ------------------------------------------------------------------
    # Standalone scan (no registry required)
    # ------------------------------------------------------------------

    def scan(self, folder_path: str, recursive: bool = True) -> ScanResult:
        """Scan *folder_path* directly, without a :class:`FolderRegistry`.

        Used by callers that just need a one-off file listing (ingest
        pipeline, sync, ingest queue) rather than persistent change tracking.
        """
        root = Path(folder_path).resolve()
        if not root.is_dir():
            logger.error("Scan target '%s' is not a directory.", root)
            return ScanResult(folder_path=str(root), files=[])

        iterator = root.rglob("*") if recursive else root.glob("*")
        files: list[ScannedFile] = []

        for entry in iterator:
            if not entry.is_file():
                continue
            ext = entry.suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue

            rel_str = str(entry.relative_to(root)).replace(os.sep, "/")
            if self._is_excluded(rel_str, self._excluded_patterns):
                continue

            try:
                stat = entry.stat()
                files.append(ScannedFile(path=entry, size_bytes=stat.st_size, mtime=stat.st_mtime))
            except OSError as exc:
                logger.warning("Could not access file '%s': %s", entry, exc)

        return ScanResult(folder_path=str(root), files=files)

    # ------------------------------------------------------------------
    # File discovery
    # ------------------------------------------------------------------

    def _discover_files(
        self,
        root: Path,
        config: FolderConfig,
    ) -> list[tuple[Path, dict[str, Any]]]:
        """Walk *root* and return ``(path, metadata)`` for matching files."""
        results: list[tuple[Path, dict[str, Any]]] = []

        if config.recursive:
            iterator = root.rglob("*")
        else:
            iterator = root.glob("*")

        for entry in iterator:
            # Skip directories.
            if not entry.is_file():
                continue

            # Check extension.
            ext = entry.suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue

            # Check exclusion patterns.
            rel = entry.relative_to(root)
            rel_str = str(rel).replace(os.sep, "/")
            if self._is_excluded(rel_str, config.excluded_patterns):
                logger.debug("Excluded: %s", entry)
                continue

            # Gather metadata.
            try:
                stat = entry.stat()
                file_hash = self._compute_hash(entry)
                metadata: dict[str, Any] = {
                    "size": stat.st_size,
                    "modified": stat.st_mtime,
                    "hash": file_hash,
                }
                results.append((entry, metadata))
            except OSError as exc:
                logger.warning("Could not access file '%s': %s", entry, exc)

        return results

    # ------------------------------------------------------------------
    # Change detection
    # ------------------------------------------------------------------

    def _detect_changes(
        self,
        folder_id: int,
        root: Path,
        files: list[tuple[Path, dict[str, Any]]],
    ) -> list[tuple[Path, tuple[str, dict[str, Any]]]]:
        """Compare *files* against registry and return changed entries.

        Returns
        -------
        list[tuple[Path, tuple[str, dict]]]
            Each entry is ``(path, (status, metadata))`` where *status* is
            ``"new"``, ``"modified"``, or ``"unchanged"``.
        """
        changed: list[tuple[Path, tuple[str, dict[str, Any]]]] = []

        for fpath, meta in files:
            rel_path = str(fpath.relative_to(root))

            existing = self._registry.get_file(folder_id, rel_path)
            if existing is None:
                # New file — register it.
                self._registry.add_file(
                    FileEntry(
                        folder_id=folder_id,
                        relative_path=rel_path,
                        file_hash=meta["hash"],
                        file_size_bytes=meta["size"],
                        last_modified=meta["modified"],
                        status="new",
                    )
                )
                changed.append((fpath, ("new", meta)))
            else:
                # Check if modified.
                if (existing.file_hash != meta["hash"]
                        or existing.last_modified < meta["modified"] - 1.0):  # 1s tolerance
                    self._registry.update_file(
                        folder_id,
                        rel_path,
                        file_hash=meta["hash"],
                        file_size_bytes=meta["size"],
                        last_modified=meta["modified"],
                        status="new",  # Re-process.
                    )
                    changed.append((fpath, ("modified", meta)))

        return changed

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_excluded(relative_path: str, patterns: list[str]) -> bool:
        """Check if *relative_path* matches any exclusion pattern."""
        for pat in patterns:
            if fnmatch.fnmatch(relative_path, pat):
                return True
            # Also check each path component.
            parts = relative_path.split("/")
            for part in parts:
                if fnmatch.fnmatch(part, pat):
                    return True
        return False

    @staticmethod
    def _compute_hash(file_path: Path, chunk_size: int = 8192) -> str | None:
        """Compute SHA-256 hash of *file_path*."""
        try:
            sha = hashlib.sha256()
            with file_path.open("rb") as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    sha.update(chunk)
            return sha.hexdigest()
        except (OSError, PermissionError) as exc:
            logger.warning("Failed to hash '%s': %s", file_path, exc)
            return None


def scan_folder(
    folder_path: str,
    recursive: bool = True,
    excluded_patterns: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Convenience wrapper to scan a directory and return file metadata.

    Parameters
    ----------
    folder_path : str
        Path to the folder to scan.
    recursive : bool
        Whether to scan subdirectories. Defaults to ``True``.
    excluded_patterns : list[str] | None
        Glob patterns to exclude. Defaults to ``None``.

    Returns
    -------
    list[dict]
        Each dict contains ``path``, ``size``, ``modified``, ``hash``, ``format``.
    """
    root = Path(folder_path).resolve()
    if not root.is_dir():
        logger.error("Scan target '%s' is not a directory.", root)
        return []

    results: list[dict[str, Any]] = []
    ext_set = SUPPORTED_EXTENSIONS

    if recursive:
        iterator = root.rglob("*")
    else:
        iterator = root.glob("*")

    for entry in iterator:
        if not entry.is_file():
            continue
        ext = entry.suffix.lower()
        if ext not in ext_set:
            continue
        rel = str(entry.relative_to(root)).replace(os.sep, "/")
        if excluded_patterns and any(fnmatch.fnmatch(rel, p) for p in excluded_patterns):
            continue
        try:
            stat = entry.stat()
            results.append({
                "path": str(entry),
                "size": stat.st_size,
                "modified": stat.st_mtime,
                "hash": FileScanner._compute_hash(entry),
                "format": _EXT_TO_DOC_FORMAT.get(ext),
            })
        except OSError:
            pass

    return results


__all__ = [
    "FileScanner",
    "ScanResult",
    "ScannedFile",
    "SUPPORTED_EXTENSIONS",
    "scan_folder",
]
