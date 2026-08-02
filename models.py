"""
SRAG Data Models (FD-08 ... FD-10).

Implements:
    FD-08: Chunk metadata schema (source, chunk_index, document_text, enriched_text).
    FD-09: Search result model with distance + optional rerank_score.
    FD-10: EnrichedMetadata model (keywords, summary, entities).

All models are dataclasses for simplicity and performance (FD-05, DD-03).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Chunk Models (FD-08)
# ---------------------------------------------------------------------------

@dataclass
class ChunkMetadata:
    """Metadata for an ingested text chunk."""
    source: str
    chunk_index: int = 0
    content_hash: str = ""
    enriched_text: Optional[str] = None


@dataclass
class Chunk:
    """A single text chunk with its metadata."""
    id: str
    source: str
    content: str
    embedding: Optional[List[float]] = None
    enriched_text: Optional[str] = None
    start_offset: Optional[int] = None
    end_offset: Optional[int] = None

    @property
    def metadata(self) -> Dict[str, Any]:
        """Return chunk metadata as a dict for storage."""
        return {
            "source": self.source,
            "chunk_index": getattr(self, "_chunk_index", 0),
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "content_hash": self.id,
            "enriched_text": self.enriched_text,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the chunk to a dict."""
        return {
            "id": self.id,
            "source": self.source,
            "content": self.content,
            "embedding": self.embedding,
            "enriched_text": self.enriched_text,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Chunk:
        """Create a Chunk from a dict."""
        return cls(
            id=data["id"],
            source=data["source"],
            content=data["content"],
            embedding=data.get("embedding"),
            enriched_text=data.get("enriched_text"),
            start_offset=data.get("start_offset"),
            end_offset=data.get("end_offset"),
        )


# ---------------------------------------------------------------------------
# Enrichment Model (FD-10)
# ---------------------------------------------------------------------------

@dataclass
class EnrichedMetadata:
    """Enriched metadata for a chunk.

    Fields:
        keywords: List of extracted keywords/phrases.
        summary: Generated summary of the chunk content.
        entities: Extracted named entities (persons, organizations, locations).
        tags: Auto-generated tags for categorization.
    """
    keywords: List[str] = field(default_factory=list)
    summary: str = ""
    entities: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps({
            "keywords": self.keywords,
            "summary": self.summary,
            "entities": self.entities,
            "tags": self.tags,
        })

    @classmethod
    def from_json(cls, json_str: str) -> EnrichedMetadata:
        """Deserialize from JSON string."""
        data = json.loads(json_str)
        return cls(
            keywords=data.get("keywords", []),
            summary=data.get("summary", ""),
            entities=data.get("entities", []),
            tags=data.get("tags", []),
        )


# ---------------------------------------------------------------------------
# Search Result Model (FD-09)
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """A single search result with distance and optional rerank score.

    Attributes:
        text: The text content of the chunk (alias for 'content').
        metadata: Dict of source metadata (source, chunk_index, etc.).
        distance: Embedding cosine/L2 distance (lower = more similar).
        rerank_score: Optional cross-encoder reranking score.
        collection: Optional ChromaDB collection name.
        _sources: Internal list of source matches for dedup transparency.
    """
    # Primary interface fields
    text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    distance: float = 0.0
    rerank_score: Optional[float] = None

    # Optional fields
    chunk_id: str = ""
    source: str = ""
    content: str = ""
    collection: str = ""
    _sources: Any = field(default=None)

    def __post_init__(self) -> None:
        """Set aliases for compatibility."""
        if not self.chunk_id and self.metadata.get("source"):
            self.chunk_id = f"{self.metadata['source']}:{self.metadata.get('chunk_index', 0)}"
        if not self.source and self.metadata.get("source"):
            self.source = self.metadata["source"]
        if not self.content and self.text:
            self.content = self.text
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        result: Dict[str, Any] = {
            "text": self.text or self.content,
            "metadata": dict(self.metadata),
            "distance": self.distance,
        }
        if self.collection:
            result["collection"] = self.collection
        if self.rerank_score is not None:
            result["rerank_score"] = self.rerank_score
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SearchResult:
        """Deserialize from dict."""
        metadata = data.get("metadata", {}) or {}
        return cls(
            text=data.get("text", ""),
            metadata=metadata,
            distance=float(data.get("distance", 0.0)),
            rerank_score=data.get("rerank_score"),
            chunk_id=data.get("chunk_id", f"{metadata.get('source', '')}:{metadata.get('chunk_index', 0)}"),
            source=metadata.get("source", data.get("source", "")),
            content=data.get("content", data.get("text", "")),
            collection=data.get("collection", ""),
        )


# ---------------------------------------------------------------------------
# ChunkWithMetadata & generate_chunk_id (FD-08, FD-10)
# ---------------------------------------------------------------------------


def generate_chunk_id(source: str, chunk_index: int, content: str) -> str:
    """Generate a deterministic chunk ID from source + index + content.

    Same inputs always produce the same ID; different inputs produce different IDs.
    """
    raw = f"{source}:{chunk_index}:{content}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


@dataclass
class ChunkWithMetadata:
    """Chunk with its full metadata (FD-08).

    Attributes:
        source: Source file / document path.
        chunk_index: Index of this chunk within the source.
        content: The text content of the chunk.
        embedding: Optional embedding vector.
        enriched_text: Optional LLM-generated enrichment.
        start_offset: Byte offset in original document (optional).
        end_offset: End byte offset (optional).
    """
    source: str
    chunk_index: int = 0
    content: str = ""
    embedding: Optional[List[float]] = None
    enriched_text: Optional[str] = None
    start_offset: Optional[int] = None
    end_offset: Optional[int] = None

    @property
    def id(self) -> str:
        """Generate a deterministic chunk ID."""
        return generate_chunk_id(self.source, self.chunk_index, self.content)


# ---------------------------------------------------------------------------
# Task Status / Ingester Models (web layer)
# ---------------------------------------------------------------------------

class TaskStatus(str, Enum):
    """Ingestion task status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Folder Model (for folders.db)
# ---------------------------------------------------------------------------

@dataclass
class Folder:
    """Represents an indexed folder."""
    id: int  # auto-increment PK
    path: str
    name: str = ""
    last_modified: float = 0.0
    chunk_count: int = 0
    enabled: bool = True

    @property
    def display_name(self) -> str:
        """Human-readable folder name."""
        return self.name or Path(self.path).name

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "id": self.id,
            "path": self.path,
            "name": self.name,
            "last_modified": self.last_modified,
            "chunk_count": self.chunk_count,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Folder:
        """Deserialize from dict."""
        return cls(
            id=data.get("id", 0),
            path=data["path"],
            name=data.get("name", ""),
            last_modified=data.get("last_modified", 0.0),
            chunk_count=data.get("chunk_count", 0),
            enabled=data.get("enabled", True),
        )


# Alias for compatibility (must be after Folder class definition)
FolderInfo = Folder


__all__ = [
    "Chunk",
    "ChunkMetadata", 
    "ChunkWithMetadata",
    "SearchResult",
    "EnrichedMetadata",
    "Folder",
    "FolderInfo",
    "TaskStatus",
    "generate_chunk_id",
]
