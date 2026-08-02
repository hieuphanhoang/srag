"""SRAG Chunker — Phase 2 implementation (DD-04).

Implements hierarchical text chunking with configurable strategies,
overlap handling, and metadata preservation.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & Data classes
# ---------------------------------------------------------------------------

class ChunkStrategy(str, Enum):
    """Text chunking strategies."""
    
    RECURSIVE = "recursive"  # Hierarchical recursive splitting
    BY_PASSAGE = "by_passage"  # Split by paragraphs/double newlines
    WORD_LEVEL = "word_level"  # Word-level with overlap


@dataclass
class Chunk:
    """A single text chunk with metadata.

    Attributes
    ----------
    content : str
        The chunked text content.
    chunk_id : str
        Unique identifier for this chunk.
    level : int
        Hierarchy level (0 = top-level, higher = sub-chunks).
    strategy : ChunkStrategy
        Strategy used to produce this chunk.
    source_file : str | None
        Original file path, if available.
    section_path : str | None
        Hierarchical section identifier (e.g., "ch1.2.3").
    metadata : dict[str, Any] = field(default_factory=dict)
        Additional per-chunk metadata.
    """

    content: str
    chunk_id: str
    level: int = 0
    strategy: ChunkStrategy = ChunkStrategy.RECURSIVE
    source_file: str | None = None
    section_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# Alias for backward compatibility / interop with codebase that expects TextChunk
TextChunk = Chunk  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Chunker configuration
# ---------------------------------------------------------------------------

@dataclass
class ChunkerConfig:
    """Configuration for the text chunker.

    Parameters
    ----------
    strategy : ChunkStrategy
        Chunking strategy to use.
    max_chunk_size : int
        Maximum tokens/characters per chunk (default *800*).
    min_chunk_size : int
        Minimum acceptable chunk size in characters (default *120*).
    overlap : int
        Overlap between adjacent chunks (default *40*).
    preserve_sections : bool
        If ``True``, attempt to respect document section boundaries.
    """

    strategy: ChunkStrategy = ChunkStrategy.RECURSIVE
    max_chunk_size: int = 800
    min_chunk_size: int = 120
    overlap: int = 40
    preserve_sections: bool = True


# ---------------------------------------------------------------------------
# Recursive chunker (hierarchical)
# ---------------------------------------------------------------------------

class _TextSplitter:
    """Internal text splitting utilities."""

    # Section heading pattern (Markdown/text style).
    SECTION_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    @classmethod
    def split_sections(cls, text: str) -> list[tuple[str, str]]:
        """Split *text* into (heading_text, body_text) tuples."""
        if not cls.SECTION_RE.search(text):
            return [("", text)]  # No sections found; single block.

        parts: list[tuple[str, str]] = []
        last_start = 0
        for match in cls.SECTION_RE.finditer(text):
            # Pre-section content (if any).
            if match.start() > last_start:
                preamble = text[last_start : match.start()].strip()
                if preamble:
                    parts.append(("", preamble))
            heading = match.group(0)
            last_start = match.end()
            parts.append((heading, ""))  # Body will be filled later.

        # Remaining content after last section.
        if last_start < len(text):
            remainder = text[last_start:].strip()
            if remainder:
                parts.append(("", remainder))

        return parts

    @classmethod
    def split_paragraphs(cls, text: str) -> list[str]:
        """Split *text* into paragraphs (double newline separated)."""
        # Normalize whitespace.
        text = text.strip()
        if not text:
            return []
        # Split on double newlines (with optional whitespace between).
        paragraphs = re.split(r"\n\s*\n", text)
        return [p.strip() for p in paragraphs if p.strip()]

    @classmethod
    def split_sentences(cls, text: str) -> list[str]:
        """Rough sentence splitter."""
        # Split on common sentence boundaries.
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]


# ---------------------------------------------------------------------------
# Public Chunker
# ---------------------------------------------------------------------------

class TextChunker:
    """Hierarchical text chunker with configurable strategies.

    Parameters
    ----------
    config : ChunkerConfig | None
        Chunking configuration.  Defaults to ``ChunkStrategy.RECURSIVE``.
    chunk_id_prefix : str
        Prefix for generated chunk IDs (default *"chunk"*).
    """

    def __init__(
        self,
        config: ChunkerConfig | None = None,
        chunk_id_prefix: str = "chunk",
    ) -> None:
        self._config = config or ChunkerConfig()
        self._id_counter = 0
        self._prefix = chunk_id_prefix

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(
        self,
        text: str,
        source_file: str | None = None,
        section_path: str | None = None,
    ) -> list[Chunk]:
        """Split *text* into chunks according to the configured strategy.

        Parameters
        ----------
        text : str
            Input text to chunk.
        source_file : str | None
            Original file path for metadata.
        section_path : str | None
            Section identifier for nested calls.

        Returns
        -------
        list[Chunk]
            List of resulting chunks.
        """
        self._id_counter += 1
        chunks: list[Chunk] = []

        if self._config.strategy == ChunkStrategy.RECURSIVE:
            chunks = self._recursive_chunk(
                text, source_file, section_path, level=0
            )
        elif self._config.strategy == ChunkStrategy.BY_PASSAGE:
            paragraphs = _TextSplitter.split_paragraphs(text)
            for para in paragraphs:
                if len(para) >= self._config.min_chunk_size:
                    chunks.append(
                        self._make_chunk(
                            para, source_file, section_path, level=0
                        )
                    )
        elif self._config.strategy == ChunkStrategy.WORD_LEVEL:
            # Word-level chunking with overlap.
            words = text.split()
            if len(words) * 5 >= self._config.min_chunk_size:  # rough token estimate
                for start in range(0, len(words), self._config.max_chunk_size - self._config.overlap):
                    end = min(start + self._config.max_chunk_size, len(words))
                    chunk_text = " ".join(words[start:end])
                    if len(chunk_text.strip()) >= self._config.min_chunk_size:
                        chunks.append(
                            self._make_chunk(
                                chunk_text, source_file, section_path, level=0
                            )
                        )

        return chunks

    def chunk_stream(
        self,
        text_iter: Iterator[str],
        source_file: str | None = None,
        section_path: str | None = None,
    ) -> list[Chunk]:
        """Consume a string iterator and chunk the accumulated text.

        Useful for processing large files incrementally.
        """
        text = "".join(text_iter)
        return self.chunk(text, source_file, section_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _recursive_chunk(
        self,
        text: str,
        source_file: str | None,
        section_path: str | None,
        level: int,
    ) -> list[Chunk]:
        """Recursively chunk *text* by sections then paragraphs."""
        chunks: list[Chunk] = []

        # Base case: text fits in one chunk.
        if len(text) <= self._config.max_chunk_size:
            stripped_len = len(text.strip())
            # min_chunk_size exists to drop tiny leftover fragments produced
            # by splitting a larger document (level > 0) — it must not also
            # discard an entire short-but-non-empty document (level 0), or
            # short files (READMEs, brief notes) silently ingest as 0 chunks.
            if stripped_len >= self._config.min_chunk_size or (level == 0 and stripped_len > 0):
                chunks.append(
                    self._make_chunk(text, source_file, section_path, level)
                )
            return chunks

        # Try splitting by sections first.
        if self._config.preserve_sections and level < 3:
            sections = _TextSplitter.split_sections(text)
            if len(sections) > 1:
                # Merge heading with following body.
                current_heading = ""
                current_body = ""
                for heading, body in sections:
                    if not current_body and not current_heading:
                        current_heading = heading
                        current_body = body
                        continue
                    combined = f"{current_heading}\n{current_body}"
                    if len(combined) > self._config.max_chunk_size:
                        # Flush current section.
                        sub_chunks = self._recursive_chunk(
                            combined, source_file, section_path, level + 1
                        )
                        chunks.extend(sub_chunks)
                        current_heading = heading
                        current_body = body
                    else:
                        current_body += f"\n\n{body}" if current_body else body

                # Flush last section.
                if current_heading or current_body:
                    combined = f"{current_heading}\n{current_body}"
                    sub_chunks = self._recursive_chunk(
                        combined, source_file, section_path, level + 1
                    )
                    chunks.extend(sub_chunks)
                return chunks

        # Fallback: split by paragraphs.
        paragraphs = _TextSplitter.split_paragraphs(text)
        if not paragraphs:
            return chunks

        # Accumulate paragraphs until chunk is full.
        accumulator = ""
        for para in paragraphs:
            candidate = f"{accumulator}\n\n{para}" if accumulator else para
            if len(candidate) > self._config.max_chunk_size and accumulator:
                # Emit accumulated chunk with overlap.
                if len(accumulator.strip()) >= self._config.min_chunk_size:
                    chunks.append(
                        self._make_chunk(
                            accumulator, source_file, section_path, level
                        )
                    )
                # Overlap: keep last portion.
                if self._config.overlap > 0:
                    accumulator = accumulator[-self._config.overlap :] + para
                else:
                    accumulator = para
            else:
                accumulator = candidate

        # Flush remaining.
        if accumulator and len(accumulator.strip()) >= self._config.min_chunk_size:
            chunks.append(
                self._make_chunk(accumulator, source_file, section_path, level)
            )

        return chunks

    def _make_chunk(
        self,
        content: str,
        source_file: str | None,
        section_path: str | None,
        level: int,
    ) -> Chunk:
        """Create a new Chunk with an incremented ID."""
        self._id_counter += 1
        chunk_id = f"{self._prefix}-{self._id_counter}"
        return Chunk(
            content=content,
            chunk_id=chunk_id,
            level=level,
            strategy=self._config.strategy,
            source_file=source_file,
            section_path=section_path,
        )

    @property
    def config(self) -> ChunkerConfig:
        """Return the current configuration."""
        return self._config


# Alias for external imports that expect 'Chunker'
Chunker = TextChunker

__all__ = [
    "Chunk",
    "ChunkStrategy",
    "ChunkerConfig",
    "TextChunker",
    "Chunker",
]
