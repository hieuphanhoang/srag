"""SRAG Ingest Pipeline — Phase 2 implementation.

Implements the core ingestion sequence: scan → convert → chunk → enrich → embed
→ upsert with cancel checkpoints (DD-08).

The pipeline is designed to be idempotent and resilient, with checkpoint-based
cancellation support for long-running operations.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from ingest.chunker import TextChunker, Chunk
from ingest.converter import DocumentConverter
from ingest.scanner import FileScanner
from llm.enricher import EnrichmentClient as Enricher
from core.embedding import Embedder
from core.models import generate_chunk_id
from store.chromadb_store import ChromaStore

logger = logging.getLogger(__name__)

# Cancel callback signature
CancelFlag = Optional[Callable[[], bool]]

# Progress callback: (files_done, files_total, current_file) -> None
ProgressCallback = Optional[Callable[[int, int, str], None]]


@dataclass
class IngestResult:
    """Result of an ingestion operation.

    Attributes:
        folder_path: The folder that was ingested.
        files_processed: Number of files successfully processed.
        chunks_created: Total number of chunks actually upserted.
        files_failed: Number of files that failed during processing.
        elapsed_sec: Wall-clock time for the ingestion.
        cancelled: Whether the ingestion was cancelled.
        errors: Human-readable per-file failure messages (DD-08 error table —
            a failed/empty conversion must surface here, never be dropped
            silently).
    """
    folder_path: str
    files_processed: int = 0
    chunks_created: int = 0
    files_failed: int = 0
    elapsed_sec: float = 0.0
    cancelled: bool = False
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for API responses."""
        return {
            "folder_path": self.folder_path,
            "files_processed": self.files_processed,
            "chunks_created": self.chunks_created,
            "files_failed": self.files_failed,
            "elapsed_sec": round(self.elapsed_sec, 3),
            "cancelled": self.cancelled,
            "errors": self.errors,
        }


class IngestCancelled(Exception):
    """Raised when the ingest pipeline is cancelled."""
    pass


def _make_doc_id(file_path: str) -> str:
    """Generate a deterministic document ID from a file path."""
    return hashlib.sha256(f"doc_{file_path}".encode("utf-8")).hexdigest()[:16]


def _checkpoint(cancel_flag: CancelFlag, checkpoint_name: str = "") -> None:
    """Check cancellation flag at a pipeline checkpoint.

    Raises:
        IngestCancelled: If the cancel flag is set.
    """
    if cancel_flag is not None and cancel_flag():
        label = f" [{checkpoint_name}]" if checkpoint_name else ""
        logger.info("Ingestion cancelled at checkpoint%s", label)
        raise IngestCancelled(f"Ingestion cancelled{label}")


def _scan_folder(folder_path: str, excluded_patterns: Optional[list[str]] = None) -> list[str]:
    """Scan a folder for ingestible files (recursive, extension-filtered)."""
    scanner = FileScanner(excluded_patterns=excluded_patterns or [])
    result = scanner.scan(folder_path)
    return [str(f.path) for f in result.files] if result else []


def run(
    folder_path: str,
    converter: DocumentConverter,
    embedder: Embedder,
    store: ChromaStore,
    collection_name: str,
    enricher: Optional[Enricher] = None,
    excluded_patterns: Optional[list[str]] = None,
    cancel_flag: CancelFlag = None,
    batch_size: int = 50,
    on_progress: ProgressCallback = None,
) -> IngestResult:
    """Run the ingest pipeline for a folder (DD-08).

    Sequence: scan → per-file convert+chunk → optional enrich → batch embed →
    upsert into *collection_name* → return result. Chunk IDs are deterministic
    (`generate_chunk_id`) so re-running this on an unchanged file upserts the
    same IDs rather than duplicating them.

    Args:
        folder_path: Path to the folder to ingest.
        converter: DocumentConverter (or ConverterManager) instance.
        embedder: Embedder instance for generating embeddings.
        store: ChromaStore to upsert chunks into.
        collection_name: Target ChromaDB collection for this folder.
        enricher: Optional Enricher instance for text enrichment.
        excluded_patterns: Glob patterns to exclude from ingestion.
        cancel_flag: Optional cancellation callback.
        batch_size: Number of chunks per embedding batch (default 50).
        on_progress: Optional callback invoked after each file is scanned.

    Returns:
        IngestResult with processing statistics.
    """
    start_time = time.time()
    folder_path = str(Path(folder_path).resolve())
    errors: list[str] = []

    files_processed = 0
    files_failed = 0

    logger.info("Scanning folder '%s' for ingestible files...", folder_path)
    file_paths = _scan_folder(folder_path, excluded_patterns)
    logger.info("Found %d files to process in '%s'", len(file_paths), folder_path)

    chunker = TextChunker()
    chunks_by_file: dict[str, list[Chunk]] = {}

    for i, file_path in enumerate(file_paths):
        _checkpoint(cancel_flag, f"file-{i + 1}/{len(file_paths)}")

        try:
            result = converter.convert(file_path)
            if not result.success or not result.text or not result.text.strip():
                msg = f"{file_path}: conversion failed or unsupported format"
                logger.warning(msg)
                errors.append(msg)
                files_failed += 1
                continue

            chunks = chunker.chunk(result.text, source_file=file_path)
            if not chunks:
                msg = f"{file_path}: converted but produced no text chunks"
                logger.warning(msg)
                errors.append(msg)
                files_failed += 1
                continue

            chunks_by_file[file_path] = chunks
            files_processed += 1
            logger.info("[%d/%d] Processed '%s' -> %d chunks",
                        i + 1, len(file_paths), os.path.basename(file_path), len(chunks))
        except Exception as e:
            files_failed += 1
            msg = f"{file_path}: {e}"
            logger.warning("File ingestion failed: %s", msg)
            errors.append(msg)
            continue
        finally:
            if on_progress is not None:
                on_progress(i + 1, len(file_paths), file_path)

    if not chunks_by_file:
        elapsed = time.time() - start_time
        return IngestResult(
            folder_path=folder_path,
            files_processed=0,
            chunks_created=0,
            files_failed=files_failed,
            elapsed_sec=elapsed,
            errors=errors,
        )

    # ── Optional enrichment ──────────────────────────────────────────
    if enricher is not None:
        texts_to_enrich: list[str] = []
        chunk_to_text_idx: dict[int, tuple[str, int]] = {}

        for file_path, chunks in chunks_by_file.items():
            for idx, chunk in enumerate(chunks):
                texts_to_enrich.append(chunk.content)
                chunk_to_text_idx[len(texts_to_enrich) - 1] = (file_path, idx)

        _checkpoint(cancel_flag, "enrichment")

        try:
            enriched_results = enricher.enrich_batch(texts_to_enrich, cancel_flag=cancel_flag)
        except Exception as e:
            logger.warning("Enrichment failed: %s. Proceeding without enrichment.", e)
            enriched_results = [None] * len(texts_to_enrich)

        for idx, enriched in enumerate(enriched_results):
            if enriched is None:
                continue
            file_path, chunk_idx = chunk_to_text_idx[idx]
            if hasattr(enriched, "to_dict"):
                chunks_by_file[file_path][chunk_idx].metadata["enriched"] = enriched.to_dict()
            elif isinstance(enriched, dict):
                chunks_by_file[file_path][chunk_idx].metadata["enriched"] = enriched
            else:
                chunks_by_file[file_path][chunk_idx].metadata["enriched"] = {"text": str(enriched)}

    # ── Batch embed ───────────────────────────────────────────────────
    all_chunks: list[tuple[str, int, Chunk]] = []
    for file_path, chunks in chunks_by_file.items():
        for idx, c in enumerate(chunks):
            all_chunks.append((file_path, idx, c))

    logger.info("Embedding %d chunk(s) in batches of %d...", len(all_chunks), batch_size)
    embeddings_result: dict[str, list[float]] = {}  # chunk_id -> embedding

    for batch_start in range(0, len(all_chunks), batch_size):
        _checkpoint(cancel_flag, f"embed-batch-{batch_start // batch_size}")

        batch_end = min(batch_start + batch_size, len(all_chunks))
        batch = all_chunks[batch_start:batch_end]
        texts = [c.content for _, _, c in batch]

        try:
            embeddings = embedder.embed_batch(texts)
        except Exception as e:
            msg = f"embedding batch starting at {batch_start} failed: {e}"
            logger.error(msg)
            errors.append(msg)
            continue  # these chunks simply won't be upserted

        for (file_path, idx, c), emb in zip(batch, embeddings):
            chunk_id = generate_chunk_id(file_path, idx, c.content)
            embeddings_result[chunk_id] = emb

    logger.info("Embedding complete. Generated %d embedding(s).", len(embeddings_result))

    # ── Upsert into ChromaDB ─────────────────────────────────────────
    _checkpoint(cancel_flag, "upsert")

    ids: list[str] = []
    documents: list[str] = []
    embeddings_list: list[list[float]] = []
    metadatas: list[dict[str, Any]] = []

    for file_path, chunks in chunks_by_file.items():
        doc_id = _make_doc_id(file_path)
        for idx, c in enumerate(chunks):
            chunk_id = generate_chunk_id(file_path, idx, c.content)
            emb = embeddings_result.get(chunk_id)
            if emb is None:
                continue  # embedding for this chunk failed; skip rather than store garbage
            ids.append(chunk_id)
            documents.append(c.content)
            embeddings_list.append(emb)
            metadatas.append({
                "filename": os.path.basename(file_path),
                "source": file_path,
                "chunk_index": idx,
                "total_chunks": len(chunks),
                "parent_document_id": doc_id,
                "enriched": c.metadata.get("enriched") if c.metadata else None,
            })

    if ids:
        try:
            store.upsert(collection_name, {
                "ids": ids,
                "documents": documents,
                "embeddings": embeddings_list,
                "metadatas": metadatas,
            })
        except Exception as e:
            logger.error("Upsert failed for collection '%s': %s", collection_name, e)
            raise

    elapsed = time.time() - start_time
    result = IngestResult(
        folder_path=folder_path,
        files_processed=files_processed,
        chunks_created=len(ids),
        files_failed=files_failed,
        elapsed_sec=elapsed,
        errors=errors,
    )

    logger.info("Ingest complete for '%s': %d files, %d chunks, %.2fs",
                folder_path, files_processed, len(ids), elapsed)

    return result


if __name__ == "__main__":
    import sys

    from core.config import load_config
    from ingest.converter import ConverterManager
    from ingest.folder_registry import collection_name_for_path
    from core.embedding import get_embedder
    from core.log import setup_logging

    if len(sys.argv) < 2:
        print("Usage: uv run python -m ingest.pipeline <folder_path>")
        raise SystemExit(1)

    target_folder = sys.argv[1]
    cfg = load_config()
    setup_logging(cfg.log_level, cfg.log_file)

    _store = ChromaStore(
        persist_directory=cfg.chromadb_path,
        tenant=cfg.chromadb_tenant,
        database=cfg.chromadb_database,
    )
    _collection = collection_name_for_path(target_folder)
    _result = run(
        folder_path=target_folder,
        converter=ConverterManager(),
        embedder=get_embedder(),
        store=_store,
        collection_name=_collection,
        batch_size=cfg.pipeline_embed_batch_size,
    )
    print(f"Processed {_result.files_processed} file(s), "
          f"{_result.chunks_created} chunk(s), {_result.files_failed} failure(s) "
          f"in {_result.elapsed_sec:.2f}s.")
    if _result.errors:
        print("Errors:")
        for err in _result.errors:
            print(f"  - {err}")
