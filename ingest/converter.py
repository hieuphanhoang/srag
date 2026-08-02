"""SRAG Ingest Converter — Phase 1 stub (DD-04, FD-21…FD-25).

File → Markdown converter using markitdown. Currently a stub;
real implementation is a Phase 2 concern.
"""

from __future__ import annotations


class FileConverter:
    """Document file converter stub."""

    def convert(self, path: str) -> tuple[str | None, dict]:
        """Convert a file to markdown text. Returns (text, metadata)."""
        # TODO: Phase 2 - implement per FD-21…FD-25 and DD-04
        return None, {}


__all__ = ["FileConverter"]