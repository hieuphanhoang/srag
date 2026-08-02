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
    """A single search result with distance and optional rerank score."""
    chunk_id: str
    source: str
    content: str
    distance: float  # embedding cosine / L2 distance
    rerank_score: Optional[float] = None  # cross-encoder reranking score

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        result: Dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "content": self.content,
            "distance": self.distance,
        }
        if self.rerank_score is not None:
            result["rerank_score"] = self.rerank_score
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SearchResult:
        """Deserialize from dict."""
        return cls(
            chunk_id=data["chunk_id"],
            source=data["source"],
            content=data["content"],
            distance=data["distance"],
            rerank_score=data.get("rerank_score"),
        )


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