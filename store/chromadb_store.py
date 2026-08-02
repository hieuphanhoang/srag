"""SRAG ChromaDB Store — Phase 1 implementation (DD-02, FD-11…FD-20).

Handles collection management, document upsertion, and vector search.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any

import chromadb

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """Single result from a ChromaDB query."""

    text: str
    metadata: dict[str, Any]
    distance: float
    collection: str
    sources: list["SearchResult"] | None = None


# ---------------------------------------------------------------------------
# ChromaStore implementation
# ---------------------------------------------------------------------------

class ChromaStore:
    """ChromaDB client wrapper with multi-collection support.

    Parameters
    ----------
    persist_directory : str
        Path to the ChromaDB persistence directory (for embedded mode).
    tenant : str | None
        Multi-tenant target.  ``None`` means default tenant.
    database : str | None
        Target database name.  ``None`` means default database.
    """

    def __init__(
        self,
        persist_directory: str = "",
        tenant: str | None = None,
        database: str | None = None,
    ) -> None:
        self._persist_dir = persist_directory
        self._tenant = tenant
        self._database = database

        # chromadb.Client(Settings(persist_directory=...)) does NOT actually
        # persist in current chromadb versions — it silently creates an
        # ephemeral in-memory client. PersistentClient is required for real
        # on-disk storage (and is what web/dependencies.get_chroma_client
        # uses, so both sides of the app see the same data).
        client_kwargs: dict[str, Any] = {}
        if self._tenant:
            client_kwargs["tenant"] = self._tenant
        if self._database:
            client_kwargs["database"] = self._database

        if self._persist_dir:
            self._client: chromadb.ClientAPI = chromadb.PersistentClient(
                path=self._persist_dir, **client_kwargs
            )
        else:
            self._client = chromadb.EphemeralClient(**client_kwargs)

        self._initialized = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Ensure the store is ready for use.

        For embedded ChromaDB this mainly creates default collections
        lazily; no network call is needed at startup time.
        """
        if self._initialized:
            return
        try:
            # Force a validation / connectivity check.
            self._client.list_collections()
        except Exception as exc:  # pragma: no cover
            logger.warning("ChromaDB initialization warning: %s", exc)
        finally:
            self._initialized = True

    def reset(self) -> None:
        """Reset the store (primarily for testing / development)."""
        if hasattr(self, "_client"):
            self._client.reset()  # type: ignore[attr-defined]
        self._initialized = False

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def get_or_create_collection(self, name: str) -> chromadb.Collection:
        """Get or create a collection by *name*.

        Uses cosine similarity as the default distance metric.
        """
        try:
            # Try to get existing collection first.
            return self._client.get_collection(name, embedding_function=None)  # type: ignore[union-attr]
        except Exception:
            pass

        # Create new collection with cosine distance (hnsw:space).
        return self._client.create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    def delete_collection(self, name: str) -> None:
        """Delete a ChromaDB collection by name."""
        try:
            self._client.delete_collection(name)
            logger.info("Deleted collection '%s'.", name)
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to delete collection '%s': %s", name, exc)

    def list_collections(self) -> list[dict[str, Any]]:
        """Return a list of dicts describing all collections."""
        collections = self._client.list_collections()  # type: ignore[attr-defined]
        return [{"name": c.name, "count": c.count()} for c in collections]

    def collection_exists(self, name: str) -> bool:
        """Check if a collection with *name* exists."""
        try:
            self._client.get_collection(name, embedding_function=None)  # type: ignore[attr-defined]
            return True
        except Exception:
            return False

    def collection_count(self, name: str) -> int:
        """Return the number of documents in *name* collection."""
        col = self.get_or_create_collection(name)
        return col.count()  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Upsert (DD-02 / FD-13 … FD-18)
    # ------------------------------------------------------------------

    def upsert(
        self,
        collection_name: str,
        data: dict[str, Any],
    ) -> None:
        """Upset documents into *collection_name*.

        Parameters
        ----------
        collection_name : str
            Target collection.
        data : dict
            Must contain keys matching ChromaDB upsert API:

            - ``ids`` (list[str]): unique document IDs
            - ``documents`` (list[str] | None): text content
            - ``embeddings`` (list[list[float]] | None): embedding vectors
            - ``metadatas`` (list[dict] | None): per-document metadata

        Raises
        ------
        ValueError
            If required fields are missing or mismatched lengths.
        """
        ids = data.get("ids")
        if not ids:
            raise ValueError("'data' must contain a non-empty 'ids' list.")

        documents = data.get("documents", [None] * len(ids))
        embeddings = data.get("embeddings", [None] * len(ids))
        metadatas = data.get("metadatas", [{}] * len(ids))

        # Validate lengths.
        sets = [ids, documents, metadatas]
        if embeddings[0] is not None:  # type: ignore[arg-type]
            sets.append(embeddings)
        lens = {len(s) for s in sets}
        if len(lens) != 1:
            raise ValueError(
                f"Mismatched lengths in upsert data: {[len(s) for s in sets]}"
            )

        # Ensure collection exists.
        col = self.get_or_create_collection(collection_name)

        try:
            col.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)  # type: ignore[arg-type]
            logger.info(
                "Upserted %d document(s) into collection '%s'.", len(ids), collection_name
            )
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"ChromaDB upsert failed for '{collection_name}': {exc}") from exc

    def delete(self, collection_name: str, ids: list[str]) -> None:
        """Delete documents by *ids* from *collection_name*."""
        col = self.get_or_create_collection(collection_name)
        try:
            col.delete(ids=ids)  # type: ignore[union-attr]
            logger.info("Deleted %d document(s) from collection '%s'.", len(ids), collection_name)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"ChromaDB delete failed for '{collection_name}': {exc}") from exc

    # ------------------------------------------------------------------
    # Vector search (DD-02 / FD-19 … FD-20)
    # ------------------------------------------------------------------

    def search(
        self,
        collection_name: str,
        query_embedding: list[float],
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Search a single ChromaDB collection.

        Parameters
        ----------
        collection_name : str
            Target collection.
        query_embedding : list[float]
            Embedding vector for the query.
        top_k : int
            Maximum number of results to return.

        Returns
        -------
        list[SearchResult]
            Sorted by ascending distance (best match first).
        """
        try:
            col = self._client.get_collection(collection_name, embedding_function=None)  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover
            logger.warning("Collection '%s' not found; returning empty results.", collection_name)
            return []

        count = col.count()  # type: ignore[return-value]
        if count == 0:
            return []

        n_results = min(top_k, max(count, 1))

        try:
            results = col.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Query failed for collection '%s': %s", collection_name, exc)
            return []

        # Parse results into SearchResult objects.
        documents = results.get("documents", [[]])[0] or []
        metadatas_list = results.get("metadatas", [[]])[0] or []
        distances = results.get("distances", [[]])[0] or []

        search_results: list[SearchResult] = []
        for doc, meta, dist in zip(documents, metadatas_list, distances):  # type: ignore[arg-type]
            search_results.append(
                SearchResult(
                    text=doc if isinstance(doc, str) else "",
                    metadata=meta if isinstance(meta, dict) else {},
                    distance=float(dist),
                    collection=collection_name,
                )
            )

        return search_results

    def multi_collection_search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        collection_names: list[str] | None = None,
    ) -> list[SearchResult]:
        """Multi-collection vector search (DD-02 / FD-19).

        Queries all registered collections and merges results.
        """
        if collection_names is None:
            # Search all collections.
            collections_info = self.list_collections()
            collection_names = [c["name"] for c in collections_info]

        all_results: list[SearchResult] = []
        for name in collection_names:
            try:
                col = self.get_or_create_collection(name)
                count = col.count()  # type: ignore[return-value]
                if count == 0:
                    continue

                n_results = min(top_k, max(count, 1))
                results = col.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    include=["documents", "metadatas", "distances"],
                )

                documents = results.get("documents", [[]])[0] or []
                metadatas_list = results.get("metadatas", [[]])[0] or []
                distances = results.get("distances", [[]])[0] or []

                for doc, meta, dist in zip(documents, metadatas_list, distances):  # type: ignore[arg-type]
                    all_results.append(
                        SearchResult(
                            text=doc if isinstance(doc, str) else "",
                            metadata=meta if isinstance(meta, dict) else {},
                            distance=float(dist),
                            collection=name,
                        )
                    )
            except Exception as exc:  # pragma: no cover
                logger.warning("Search failed for collection '%s': %s", name, exc)

        # Global sort by distance (ascending).
        all_results.sort(key=lambda r: r.distance)
        return all_results[:top_k]


    def count(self) -> int:
        """Return the total number of documents across all collections."""
        total = 0
        for col_info in self.list_collections():
            total += col_info.get("count", 0)
        return total

    @staticmethod
    def merge_results(
        results: list[SearchResult],
        top_k: int | None = None,
        deduplicate: bool = True,
    ) -> list[SearchResult]:
        """Merge and re-rank multi-source *results* (FD-20 / DD-03).

        Parameters
        ----------
        results : list[SearchResult]
            Raw search results from one or more sources.
        top_k : int | None
            Maximum number of results to return.  If ``None``, returns all.
        deduplicate : bool
            If ``True``, remove duplicate documents by ID (first occurrence wins).

        Returns
        -------
        list[SearchResult]
            Merged results sorted by ascending distance.
        """
        if deduplicate:
            seen_ids: set[str] = set()
            unique: list[SearchResult] = []
            for r in results:
                # Build a lightweight key from metadata + text hash.
                key = (r.collection, r.text)
                if key not in seen_ids:
                    seen_ids.add(key)
                    unique.append(r)
            results = unique

        results = sorted(results, key=lambda r: r.distance)

        if top_k is not None:
            return results[:top_k]
        return results

    def get_by_id(self, doc_id: str, collection_name: str | None = None) -> dict[str, Any] | None:
        """Retrieve a document by its ID."""
        collections_to_search = [collection_name] if collection_name else self.list_collections()
        for col_info in collections_to_search:
            col_name = col_info["name"] if isinstance(col_info, dict) else col_info
            try:
                col = self.get_or_create_collection(col_name)
                result = col.get(ids=[doc_id])  # type: ignore[union-attr]
                if result and result.get("ids") and len(result["ids"]) > 0:
                    return {
                        "id": result["ids"][0],
                        "text": result["documents"][0] if result.get("documents") else "",
                        "metadata": result["metadatas"][0] if result.get("metadatas") else {},
                    }
            except Exception:
                continue
        return None

    def __enter__(self) -> "ChromaStore":
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, *args) -> None:
        """Context manager exit."""
        pass


__all__ = ["ChromaStore", "SearchResult"]
