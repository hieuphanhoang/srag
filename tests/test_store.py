"""Tests for store/chromadb_store.py."""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from store.chromadb_store import ChromaStore, SearchResult


# ------------------------------------------------------------------ #
# SearchResult (pure dataclass – no real Chroma dependency needed)    #
# ------------------------------------------------------------------ #

class TestSearchResult:
    def test_search_result_creation(self):
        sr = SearchResult(
            text="Hello world",
            metadata={"source": "test.txt"},
            distance=0.5,
            collection="test_col",
        )
        assert sr.text == "Hello world"
        assert sr.distance == 0.5

    def test_search_result_with_sources(self):
        sr = SearchResult(
            text="Hi", metadata={}, distance=0.1, collection="c", sources=[]
        )
        assert sr.sources == []


# ------------------------------------------------------------------ #
# ChromaStore — all tests use mocking to avoid the global Chroma      #
# ephemeral singleton that crashes when multiple stores are created.  #
# ------------------------------------------------------------------ #

def _make_mock_store(persist_dir: str = "/tmp/fake_chroma") -> tuple[ChromaStore, MagicMock]:
    """Return (store, mock_client) with minimal mocking."""
    with patch("chromadb.PersistentClient") as mock_cls:
        mock_client = MagicMock()
        mock_client.list_collections.return_value = []
        mock_cls.return_value = mock_client

        store = ChromaStore(persist_directory=persist_dir)
        return store, mock_client


class TestChromaStoreInitialization:
    def test_default_initialization(self):
        store, _ = _make_mock_store()
        assert store is not None

    def test_with_persist_directory(self):
        with patch("chromadb.PersistentClient"):
            store = ChromaStore(persist_directory="/tmp/test_chroma")
            assert store._persist_dir == "/tmp/test_chroma"


class TestChromaStoreLifecycle:
    def test_initialize(self):
        store, _ = _make_mock_store()
        store.initialize()
        assert store._initialized is True

    def test_reset(self):
        store, _ = _make_mock_store()
        store.initialize()
        store.reset()
        assert store._initialized is False


class TestChromaStoreCollectionManagement:
    """All ChromaDB calls are mocked — no real database involved."""

    @pytest.fixture
    def mock_col(self):
        col = MagicMock()
        type(col).metadata = PropertyMock(return_value={"hnsw:space": "cosine"})
        return col

    @pytest.fixture
    def store_with_mock_client(self, mock_col):
        with patch("chromadb.PersistentClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.list_collections.return_value = []
            mock_client.get_collection.side_effect = Exception("not found")
            mock_client.create_collection.return_value = mock_col
            mock_cls.return_value = mock_client

            store = ChromaStore(persist_directory="/tmp/test_chroma")
            store.initialize()
            yield store, mock_client, mock_col

    def test_get_or_create_collection_new(self, store_with_mock_client):
        store, mock_client, _ = store_with_mock_client
        col = store.get_or_create_collection("new_col")
        assert col is not None
        mock_client.create_collection.assert_called()

    def test_get_or_create_collection_existing(self, store_with_mock_client):
        store, mock_client, mock_col = store_with_mock_client
        # Make get_collection succeed.
        mock_client.get_collection.side_effect = None
        mock_client.get_collection.return_value = mock_col
        mock_client.create_collection.reset_mock()

        col1 = store.get_or_create_collection("existing")
        col2 = store.get_or_create_collection("existing")
        assert col1 is col2
        # create_collection should NOT be called for existing.
        mock_client.create_collection.assert_not_called()

    def test_delete_collection(self, store_with_mock_client):
        store, mock_client, _ = store_with_mock_client
        store.delete_collection("test_col")
        mock_client.delete_collection.assert_called_once_with("test_col")

    def test_list_collections(self, store_with_mock_client):
        store, mock_client, _ = store_with_mock_client
        mock_col1 = MagicMock()
        mock_col1.name = "col1"
        mock_col1.count.return_value = 10
        mock_col2 = MagicMock()
        mock_col2.name = "col2"
        mock_col2.count.return_value = 5
        mock_client.list_collections.return_value = [mock_col1, mock_col2]

        result = store.list_collections()
        assert len(result) == 2
        assert result[0]["name"] == "col1"
        assert result[0]["count"] == 10

    def test_collection_exists_true(self, store_with_mock_client):
        store, mock_client, mock_col = store_with_mock_client
        mock_client.get_collection.side_effect = None
        mock_client.get_collection.return_value = mock_col
        assert store.collection_exists("exists") is True

    def test_collection_exists_false(self, store_with_mock_client):
        store, mock_client, _ = store_with_mock_client
        mock_client.get_collection.side_effect = Exception("not found")
        assert store.collection_exists("missing") is False


class TestChromaStoreUpsert:
    @pytest.fixture
    def mock_store(self):
        with patch("chromadb.PersistentClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.list_collections.return_value = []
            mock_cls.return_value = mock_client

            mock_col = MagicMock()
            type(mock_col).metadata = PropertyMock(return_value={"hnsw:space": "cosine"})
            mock_client.create_collection.return_value = mock_col
            mock_client.get_collection.return_value = mock_col

            store = ChromaStore(persist_directory="/tmp/test_chroma")
            store.initialize()
            store._client = mock_client
            yield store, mock_col

    def test_upsert_valid_data(self, mock_store):
        store, _ = mock_store
        data = {
            "ids": ["doc1", "doc2"],
            "documents": ["Hello", "World"],
            "metadatas": [{"source": "a.txt"}, {"source": "b.txt"}],
            "embeddings": [[0.1] * 384, [0.2] * 384],
        }
        # Should not raise.
        store.upsert("test_col", data)

    def test_upsert_empty_ids(self, mock_store):
        store, _ = mock_store
        with pytest.raises(ValueError, match="must contain a non-empty"):
            store.upsert("test_col", {"ids": []})

    def test_upsert_missing_ids(self, mock_store):
        store, _ = mock_store
        with pytest.raises(ValueError, match="must contain a non-empty"):
            store.upsert("test_col", {})

    def test_upsert_mismatched_lengths(self, mock_store):
        store, _ = mock_store
        data = {
            "ids": ["doc1"],
            "documents": ["Hello", "World"],  # mismatch
            "metadatas": [{"source": "a.txt"}],
        }
        with pytest.raises(ValueError, match="Mismatched lengths"):
            store.upsert("test_col", data)


class TestChromaStoreDelete:
    @pytest.fixture
    def mock_store(self):
        with patch("chromadb.PersistentClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.list_collections.return_value = []
            mock_cls.return_value = mock_client

            mock_col = MagicMock()
            type(mock_col).metadata = PropertyMock(return_value={"hnsw:space": "cosine"})
            mock_client.create_collection.return_value = mock_col
            mock_client.get_collection.return_value = mock_col

            store = ChromaStore(persist_directory="/tmp/test_chroma")
            store.initialize()
            store._client = mock_client
            yield store, mock_col

    def test_delete(self, mock_store):
        store, mock_col = mock_store
        store.delete("test_col", ["doc1", "doc2"])
        mock_col.delete.assert_called_once_with(ids=["doc1", "doc2"])


class TestChromaStoreSearch:
    @pytest.fixture
    def mock_search_store(self):
        with patch("chromadb.PersistentClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.list_collections.return_value = []
            mock_cls.return_value = mock_client

            mock_col = MagicMock()
            type(mock_col).metadata = PropertyMock(return_value={"hnsw:space": "cosine"})
            # Mock query results.
            mock_query_result = {
                "documents": [["doc1", "doc2"]],
                "distances": [[0.1, 0.3]],
                "metadatas": [[{"source": "a.txt"}, {"source": "b.txt"}]],
            }
            mock_col.query.return_value = mock_query_result
            mock_col.count.return_value = 2

            mock_client.create_collection.return_value = mock_col
            mock_client.get_collection.return_value = mock_col

            store = ChromaStore(persist_directory="/tmp/test_chroma")
            store.initialize()
            store._client = mock_client
            yield store, mock_col

    def test_search_returns_results(self, mock_search_store):
        store, _ = mock_search_store
        results = store.search("test_col", query_embedding=[0.1] * 384, top_k=2)
        assert len(results) == 2
        assert results[0].text == "doc1"
        assert results[0].distance == 0.1

    def test_search_empty_collection(self):
        """Search on a collection with 0 docs should return empty."""
        with patch("chromadb.PersistentClient") as mock_cls:
            mock_client = MagicMock()
            mock_col = MagicMock()
            mock_col.count.return_value = 0
            mock_client.get_collection.return_value = mock_col
            mock_client.list_collections.return_value = []
            mock_cls.return_value = mock_client

            store = ChromaStore(persist_directory="/tmp/test_chroma")
            store.initialize()
            store._client = mock_client
            results = store.search("empty_col", query_embedding=[0.1] * 384, top_k=2)
            assert results == []


class TestChromaStoreMultiCollectionSearch:
    @pytest.fixture
    def mock_multi_store(self):
        with patch("chromadb.PersistentClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.list_collections.return_value = []
            mock_cls.return_value = mock_client

            # Create two mock collections.
            mock_col1 = MagicMock()
            mock_col1.count.return_value = 2
            mock_query_result1 = {
                "documents": [["doc1"]],
                "distances": [[0.1]],
                "metadatas": [[{"source": "a.txt"}]],
            }
            mock_col1.query.return_value = mock_query_result1
            type(mock_col1).metadata = PropertyMock(return_value={"hnsw:space": "cosine"})

            mock_col2 = MagicMock()
            mock_col2.count.return_value = 1
            mock_query_result2 = {
                "documents": [["doc2"]],
                "distances": [[0.3]],
                "metadatas": [[{"source": "b.txt"}]],
            }
            mock_col2.query.return_value = mock_query_result2
            type(mock_col2).metadata = PropertyMock(return_value={"hnsw:space": "cosine"})

            def get_or_create(name):
                return mock_col1 if name == "col1" else mock_col2
            mock_client.get_collection.side_effect = lambda n, **kw: get_or_create(n)
            mock_client.create_collection.side_effect = lambda n, **kw: get_or_create(n)

            store = ChromaStore(persist_directory="/tmp/test_chroma")
            store.initialize()
            store._client = mock_client
            yield store

    def test_multi_collection_search(self, mock_multi_store):
        store = mock_multi_store
        results = store.multi_collection_search(
            query_embedding=[0.1] * 384,
            top_k=5,
            collection_names=["col1", "col2"],
        )
        assert len(results) == 2
        # Results should be sorted by distance ascending.
        assert results[0].distance <= results[1].distance


class TestChromaStoreContextManager:
    def test_context_manager_exists(self):
        """Verify __enter__ and __exit__ are defined (interface check)."""
        with patch("chromadb.PersistentClient"):
            store = ChromaStore(persist_directory="/tmp/test_cm")
            assert hasattr(store, "__enter__")
            assert hasattr(store, "__exit__")


class TestChromaStoreMergeResults:
    """Test the static merge_results helper."""

    def test_merge_sorts_by_distance(self):
        results = [
            SearchResult(text="c", metadata={}, distance=0.9, collection="x"),
            SearchResult(text="a", metadata={}, distance=0.1, collection="x"),
            SearchResult(text="b", metadata={}, distance=0.5, collection="x"),
        ]
        merged = ChromaStore.merge_results(results, top_k=2)
        assert len(merged) == 2
        assert merged[0].text == "a"
        assert merged[1].text == "b"

    def test_merge_empty_list(self):
        results = ChromaStore.merge_results([], top_k=5)
        assert results == []


class TestChromaStoreErrorHandling:
    def test_nonexistent_persist_directory(self):
        """Should not crash on init even if directory doesn't exist."""
        with patch("chromadb.PersistentClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.list_collections.return_value = []
            mock_cls.return_value = mock_client

            store = ChromaStore(persist_directory="/nonexistent/path/fake")
            assert store is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])