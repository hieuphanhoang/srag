"""Tests for models.py — covers FD-08, FD-09, FD-10."""

import json
from unittest.mock import patch

import pytest

# Import directly from the module (no srag package prefix needed)
from models import Chunk, EnrichedMetadata, Folder, SearchResult


class TestChunkModel:
    def test_chunk_creation(self):
        """Chunk dataclass can be instantiated with required fields."""
        c = Chunk(id="ch-1", source="test.txt", content="Hello world")
        assert c.id == "ch-1"
        assert c.source == "test.txt"
        assert c.content == "Hello world"

    def test_chunk_with_embedding(self):
        """Chunk can have an optional embedding."""
        emb = [0.1, 0.2, 0.3]
        c = Chunk(id="ch-2", source="test.txt", content="Hello", embedding=emb)
        assert c.embedding == emb

    def test_chunk_metadata(self):
        """Chunk.metadata returns a dict with expected keys."""
        c = Chunk(id="ch-3", source="doc.pdf", content="text", start_offset=0, end_offset=4)
        # Set _chunk_index as attribute since metadata uses getattr
        object.__setattr__(c, "_chunk_index", 5)
        meta = c.metadata
        assert "source" in meta
        assert meta["source"] == "doc.pdf"
        assert "chunk_index" in meta

    def test_chunk_to_dict(self):
        """Chunk.to_dict serializes all fields."""
        c = Chunk(id="ch-4", source="s.txt", content="hi", start_offset=0, end_offset=2)
        d = c.to_dict()
        assert d["id"] == "ch-4"
        assert d["source"] == "s.txt"
        assert d["content"] == "hi"

    def test_chunk_from_dict(self):
        """Chunk.from_dict deserializes correctly."""
        data = {
            "id": "ch-5",
            "source": "s.txt",
            "content": "hello world",
            "embedding": [0.1, 0.2],
            "enriched_text": None,
            "start_offset": 0,
            "end_offset": 5,
        }
        c = Chunk.from_dict(data)
        assert c.id == "ch-5"
        assert c.content == "hello world"
        assert c.embedding == [0.1, 0.2]


class TestSearchResultModel:
    def test_search_result_creation(self):
        """SearchResult can be instantiated."""
        sr = SearchResult(text="Hello", distance=0.5)
        assert sr.text == "Hello"
        assert sr.distance == 0.5

    def test_search_result_to_dict(self):
        """SearchResult.to_dict serializes correctly."""
        sr = SearchResult(text="Hi", distance=0.3, collection="test_col")
        d = sr.to_dict()
        assert d["text"] == "Hi"
        assert d["distance"] == 0.3
        assert d["collection"] == "test_col"

    def test_search_result_from_dict(self):
        """SearchResult.from_dict deserializes correctly."""
        data = {"text": "Hello", "metadata": {"source": "s.txt"}, "distance": 0.1}
        sr = SearchResult.from_dict(data)
        assert sr.text == "Hello"
        assert sr.distance == 0.1

    def test_search_result_with_rerank(self):
        """SearchResult can have a rerank_score."""
        sr = SearchResult(text="Hi", distance=0.5, rerank_score=0.9)
        d = sr.to_dict()
        assert d["rerank_score"] == 0.9


class TestEnrichedMetadata:
    def test_enriched_metadata_default(self):
        """EnrichedMetadata defaults to empty lists/strings."""
        em = EnrichedMetadata()
        assert em.keywords == []
        assert em.summary == ""
        assert em.entities == []
        assert em.tags == []

    def test_enriched_metadata_to_json_roundtrip(self):
        """EnrichedMetadata serializes and deserializes correctly."""
        original = EnrichedMetadata(
            keywords=["hello", "world"],
            summary="A summary",
            entities=["Alice"],
            tags=["test"],
        )
        json_str = original.to_json()
        restored = EnrichedMetadata.from_json(json_str)
        assert restored.keywords == ["hello", "world"]
        assert restored.summary == "A summary"
        assert restored.entities == ["Alice"]
        assert restored.tags == ["test"]


class TestFolderModel:
    def test_folder_creation(self):
        """Folder dataclass can be instantiated."""
        f = Folder(id=1, path="/tmp/docs", name="docs")
        assert f.id == 1
        assert f.path == "/tmp/docs"
        assert f.name == "docs"

    def test_folder_display_name(self):
        """Folder.display_name returns the name or path basename."""
        f = Folder(id=1, path="/tmp/unknown", name="")
        # Should return the basename of the path when name is empty
        from pathlib import Path
        expected = Path("/tmp/unknown").name
        assert f.display_name == expected

    def test_folder_to_dict_roundtrip(self):
        """Folder serializes and deserializes correctly."""
        orig = Folder(id=42, path="/data", name="data", last_modified=100.0, chunk_count=5, enabled=False)
        d = orig.to_dict()
        restored = Folder.from_dict(d)
        assert restored.id == 42
        assert restored.path == "/data"
        assert restored.enabled is False
        assert restored.chunk_count == 5


class TestChunkWithMetadata:
    """Tests for Chunk metadata integration."""

    def test_chunk_metadata_source(self):
        """Chunk.metadata includes source field."""
        c = Chunk(id="ch-1", source="/path/to/file.txt", content="content")
        meta = c.metadata
        assert "source" in meta
        assert meta["source"] == "/path/to/file.txt"

    def test_chunk_metadata_enriched_text(self):
        """Chunk.metadata includes enriched_text field."""
        c = Chunk(id="ch-1", source="/path/file.txt", content="content", enriched_text="enriched")
        meta = c.metadata
        assert "enriched_text" in meta

    def test_chunk_metadata_content_hash(self):
        """Chunk.metadata includes content_hash field."""
        c = Chunk(id="hash123", source="/path/file.txt", content="content")
        meta = c.metadata
        assert "content_hash" in meta


class TestSearchResultIntegration:
    """Integration tests for SearchResult."""

    def test_search_result_aliases(self):
        """SearchResult sets aliases from metadata."""
        sr = SearchResult(
            text="Hello world",
            distance=0.3,
            metadata={"source": "test.txt", "chunk_index": 0},
        )
        assert sr.source == "test.txt"
        assert sr.content == "Hello world"

    def test_search_result_from_dict_aliases(self):
        """SearchResult.from_dict sets aliases."""
        data = {
            "text": "Hello",
            "metadata": {"source": "doc.txt", "chunk_index": 3},
            "distance": 0.1,
        }
        sr = SearchResult.from_dict(data)
        assert sr.source == "doc.txt"
        assert sr.chunk_id is not None


class TestFolderIntegration:
    """Tests for Folder model."""

    def test_folder_defaults(self):
        """Folder has sensible defaults."""
        f = Folder(id=0, path="/test")
        assert f.name == ""
        assert f.last_modified == 0.0
        assert f.chunk_count == 0
        assert f.enabled is True