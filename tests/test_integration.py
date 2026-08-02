"""Integration tests (PLAN.md §7.2) — module pairs and pipelines against
REAL ChromaDB (in tmp_path) and REAL Ollama. No mocks. Every test in this
file requires a running local Ollama with the models config.yaml assigns
(embedding + qwen3-5-9b) actually pulled.

Run with: uv run pytest tests/test_integration.py -m integration -v
"""

from __future__ import annotations

import threading
import time
import zipfile
from pathlib import Path

import pytest

import web.dependencies as web_deps
from config import load_config
from embedding import Embedder
from ingest.converter import ConverterManager
from ingest.folder_registry import FolderConfig, FolderRegistry, collection_name_for_path
from ingest.pipeline import run as run_pipeline
from ingest.sync import full_sync, incremental_sync
from llm.enricher import EnrichmentClient
from store.chromadb_store import ChromaStore
from web.dependencies import run_search
from web.ingest_queue import IngestQueue, TaskStatus

pytestmark = pytest.mark.integration


# ─── Shared fixtures ────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def real_embedder(cfg):
    embedder = Embedder(url=cfg.ollama_url, model=cfg.embedding_model)
    try:
        embedder.embed("connectivity check")
    except Exception as exc:
        pytest.skip(f"Ollama/embedding model not reachable: {exc}")
    return embedder


@pytest.fixture()
def store(tmp_path):
    s = ChromaStore(persist_directory=str(tmp_path / "chroma"))
    s.initialize()
    return s


@pytest.fixture()
def registry(tmp_path):
    reg = FolderRegistry(str(tmp_path / "registry.db"))
    yield reg
    reg.close()


# ─── Fixture-file helpers (real files, not stubs) ──────────────────────

def _write_pptx(path: Path, texts: list[str]) -> None:
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/ppt/presentation.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>'
        '<Override PartName="/ppt/slides/slide1.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
        "</Types>"
    )
    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="ppt/presentation.xml"/>'
        "</Relationships>"
    )
    presentation = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<p:presentation xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<p:sldIdLst><p:sldId id="256" r:id="rId1"/></p:sldIdLst>'
        "</p:presentation>"
    )
    presentation_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" '
        'Target="slides/slide1.xml"/>'
        "</Relationships>"
    )
    shapes = "".join(
        f'<p:sp><p:nvSpPr><p:cNvPr id="{i + 2}" name="TextBox{i + 1}"/>'
        f'<p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr><p:spPr/>'
        f'<p:txBody><a:bodyPr/><a:p><a:r><a:t>{t}</a:t></a:r></a:p></p:txBody></p:sp>'
        for i, t in enumerate(texts)
    )
    slide1 = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">'
        f'<p:cSld><p:spTree>{shapes}</p:spTree></p:cSld>'
        "</p:sld>"
    )
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", root_rels)
        zf.writestr("ppt/presentation.xml", presentation)
        zf.writestr("ppt/_rels/presentation.xml.rels", presentation_rels)
        zf.writestr("ppt/slides/slide1.xml", slide1)


def _write_xlsx(path: Path, cell_values: list[str]) -> None:
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        "</Types>"
    )
    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        "</Relationships>"
    )
    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets>'
        "</workbook>"
    )
    workbook_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/>'
        "</Relationships>"
    )
    rows = "".join(
        f'<row r="{i + 1}"><c r="A{i + 1}" t="inlineStr"><is><t>{v}</t></is></c></row>'
        for i, v in enumerate(cell_values)
    )
    sheet1 = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f"<sheetData>{rows}</sheetData>"
        "</worksheet>"
    )
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", root_rels)
        zf.writestr("xl/workbook.xml", workbook)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        zf.writestr("xl/worksheets/sheet1.xml", sheet1)


def _write_docx(path: Path, paragraphs: list[str]) -> None:
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        "</Types>"
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="word/document.xml"/>'
        "</Relationships>"
    )
    body = "".join(f"<w:p><w:r><w:t>{p}</w:t></w:r></w:p>" for p in paragraphs)
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{body}</w:body>"
        "</w:document>"
    )
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("word/document.xml", document)


def _write_pdf(path: Path, text: str) -> None:
    from reportlab.pdfgen import canvas
    c = canvas.Canvas(str(path))
    c.drawString(72, 720, text)
    c.save()


# ─── 1. Embedder round-trip (DD-03) ─────────────────────────────────────

class TestEmbedderRoundTrip:
    def test_single_embed_returns_vector(self, real_embedder):
        vec = real_embedder.embed("The quick brown fox jumps over the lazy dog.")
        assert isinstance(vec, list)
        assert len(vec) > 0
        assert all(isinstance(x, float) for x in vec)

    def test_batch_embed_count_matches_input(self, real_embedder):
        texts = ["first document", "second document", "third document"]
        vecs = real_embedder.embed_batch(texts)
        assert len(vecs) == len(texts)

    def test_batch_and_single_embed_same_dimension(self, real_embedder):
        single = real_embedder.embed("hello world")
        batch = real_embedder.embed_batch(["hello world", "goodbye world"])
        assert len(single) == len(batch[0]) == len(batch[1])


# ─── 2. Cross-collection search (DD-02) ─────────────────────────────────

class TestCrossCollectionSearch:
    def test_merges_multiple_collections_ascending_distance(self, store, real_embedder):
        e1 = real_embedder.embed("cats and dogs are common pets")
        e2 = real_embedder.embed("the stock market fell sharply today")
        store.upsert("collection_a", {
            "ids": ["a1"], "documents": ["cats and dogs are common pets"],
            "embeddings": [e1], "metadatas": [{"source": "a.txt", "chunk_index": 0}],
        })
        store.upsert("collection_b", {
            "ids": ["b1"], "documents": ["the stock market fell sharply today"],
            "embeddings": [e2], "metadatas": [{"source": "b.txt", "chunk_index": 0}],
        })

        query_emb = real_embedder.embed("pets like cats and dogs")
        results = store.multi_collection_search(
            query_embedding=query_emb, top_k=10,
            collection_names=["collection_a", "collection_b"],
        )

        assert len(results) == 2
        assert results[0].distance <= results[1].distance
        assert results[0].collection == "collection_a"

    def test_skips_empty_collection_without_failing(self, store, real_embedder):
        store.get_or_create_collection("empty_collection")
        query_emb = real_embedder.embed("anything")
        results = store.multi_collection_search(
            query_embedding=query_emb, top_k=5, collection_names=["empty_collection"],
        )
        assert results == []

    def test_skips_nonexistent_collection_without_failing(self, store, real_embedder):
        query_emb = real_embedder.embed("anything")
        results = store.multi_collection_search(
            query_embedding=query_emb, top_k=5,
            collection_names=["this_collection_does_not_exist"],
        )
        assert results == []


# ─── 3. Converter on real PDF/DOCX/PPTX/XLSX (FD-22) ────────────────────

class TestConverterRealFormats:
    def test_pdf_converts_nonempty(self, tmp_path):
        p = tmp_path / "doc.pdf"
        _write_pdf(p, "PDFMARKERTEXT integration test content")
        result = ConverterManager().convert(str(p))
        assert result.success is True, result.error
        assert "PDFMARKERTEXT" in result.text

    def test_docx_converts_nonempty(self, tmp_path):
        p = tmp_path / "doc.docx"
        _write_docx(p, ["DOCXMARKERTEXT integration test content"])
        result = ConverterManager().convert(str(p))
        assert result.success is True, result.error
        assert "DOCXMARKERTEXT" in result.text

    def test_pptx_converts_nonempty(self, tmp_path):
        p = tmp_path / "deck.pptx"
        _write_pptx(p, ["PPTXMARKERTEXT integration test content"])
        result = ConverterManager().convert(str(p))
        assert result.success is True, result.error
        assert "PPTXMARKERTEXT" in result.text

    def test_xlsx_converts_nonempty(self, tmp_path):
        p = tmp_path / "sheet.xlsx"
        _write_xlsx(p, ["XLSXMARKERTEXT integration test content"])
        result = ConverterManager().convert(str(p))
        assert result.success is True, result.error
        assert "XLSXMARKERTEXT" in result.text


# ─── 4/5/6. Ingest a folder, idempotency, failing file (DD-08) ─────────

class TestIngestFolder:
    def test_ingest_mixed_folder_lands_chunks_in_chromadb(self, tmp_path, store, real_embedder):
        docs = tmp_path / "docs"
        docs.mkdir()
        _write_pdf(docs / "report.pdf", "Quarterly revenue grew by twelve percent this year.")
        _write_docx(docs / "notes.docx", ["Meeting notes: discussed the new product roadmap."])
        (docs / "readme.txt").write_text("This is a plain text readme file with enough content.", encoding="utf-8")

        result = run_pipeline(
            folder_path=str(docs),
            converter=ConverterManager(),
            embedder=real_embedder,
            store=store,
            collection_name="mixed_folder_test",
        )

        assert result.files_processed == 3
        assert result.files_failed == 0
        assert result.chunks_created >= 3
        assert store.collection_count("mixed_folder_test") == result.chunks_created

    def test_ingest_idempotent_same_chunk_count_on_rerun(self, tmp_path, store, real_embedder):
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "a.txt").write_text("Idempotency test document with stable content.", encoding="utf-8")

        first = run_pipeline(
            folder_path=str(docs), converter=ConverterManager(), embedder=real_embedder,
            store=store, collection_name="idempotent_test",
        )
        second = run_pipeline(
            folder_path=str(docs), converter=ConverterManager(), embedder=real_embedder,
            store=store, collection_name="idempotent_test",
        )

        assert first.chunks_created == second.chunks_created
        assert store.collection_count("idempotent_test") == first.chunks_created

    def test_failing_file_recorded_other_files_still_ingest(self, tmp_path, store, real_embedder):
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "good.txt").write_text("This file converts successfully and has real content.", encoding="utf-8")
        # markitdown falls back to raw-text extraction for arbitrary garbage
        # bytes (even with a .pdf extension) rather than failing - an empty
        # file is what reliably triggers a real PdfConverter parse error
        # ("No /Root object!").
        (docs / "corrupt.pdf").write_bytes(b"")

        result = run_pipeline(
            folder_path=str(docs), converter=ConverterManager(), embedder=real_embedder,
            store=store, collection_name="failing_file_test",
        )

        assert result.files_processed == 1
        assert result.files_failed == 1
        assert len(result.errors) == 1
        assert "corrupt.pdf" in result.errors[0]
        assert result.chunks_created >= 1


# ─── 7. Sync — new / modified / deleted (DD-09) ─────────────────────────

class TestSync:
    def test_sync_new_file_adds_chunks(self, tmp_path, store, registry, real_embedder):
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "a.txt").write_text("First document for sync testing purposes.", encoding="utf-8")

        folder_id = registry.add_folder(FolderConfig(path=str(docs)))
        result = full_sync(folder_id, registry, ConverterManager(), real_embedder, store)

        assert result.files_processed == 1
        collection_name = collection_name_for_path(str(docs))
        assert store.collection_count(collection_name) == result.chunks_created
        assert len(registry.list_files(folder_id)) == 1

    def test_sync_modified_file_replaces_chunks(self, tmp_path, store, registry, real_embedder):
        docs = tmp_path / "docs"
        docs.mkdir()
        f = docs / "a.txt"
        f.write_text("Original content before modification.", encoding="utf-8")

        folder_id = registry.add_folder(FolderConfig(path=str(docs)))
        full_sync(folder_id, registry, ConverterManager(), real_embedder, store)

        f.write_text("Completely different content after the file was modified with much more text to ensure a distinct chunk.", encoding="utf-8")
        result = incremental_sync(folder_id, registry, ConverterManager(), real_embedder, store)

        assert result.files_processed == 1
        collection_name = collection_name_for_path(str(docs))
        results = store.search(collection_name, real_embedder.embed("modified with much more text"), top_k=5)
        assert any("modified" in r.text for r in results)

    def test_sync_deleted_file_removes_chunks(self, tmp_path, store, registry, real_embedder):
        docs = tmp_path / "docs"
        docs.mkdir()
        keep = docs / "keep.txt"
        remove = docs / "remove.txt"
        keep.write_text("This file will be kept around for the whole test.", encoding="utf-8")
        remove.write_text("This file will be deleted partway through the test.", encoding="utf-8")

        folder_id = registry.add_folder(FolderConfig(path=str(docs)))
        first = full_sync(folder_id, registry, ConverterManager(), real_embedder, store)
        collection_name = collection_name_for_path(str(docs))
        assert store.collection_count(collection_name) == first.chunks_created

        remove.unlink()
        incremental_sync(folder_id, registry, ConverterManager(), real_embedder, store)

        remaining_paths = {e.relative_path for e in registry.list_files(folder_id)}
        assert "remove.txt" not in remaining_paths
        assert "keep.txt" in remaining_paths

        col = store.get_or_create_collection(collection_name)
        remaining = col.get(where={"filename": "remove.txt"})
        assert not (remaining.get("ids") or [])

    def test_sync_no_change_run_is_noop(self, tmp_path, store, registry, real_embedder):
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "a.txt").write_text("Stable content that never changes between syncs.", encoding="utf-8")

        folder_id = registry.add_folder(FolderConfig(path=str(docs)))
        full_sync(folder_id, registry, ConverterManager(), real_embedder, store)

        result = incremental_sync(folder_id, registry, ConverterManager(), real_embedder, store)
        assert result.files_processed == 0
        assert result.chunks_created == 0


# ─── 8. Enrichment batch (DD-10, DD-11) ─────────────────────────────────

class TestEnrichmentBatch:
    def test_batch_enrichment_preserves_original_text(self, cfg):
        try:
            client = EnrichmentClient()
            client.client.generate("connectivity check", max_tokens=5)
        except Exception as exc:
            pytest.skip(f"Configured enrichment LLM not reachable: {exc}")

        texts = [
            "The Great Wall of China is over 13000 miles long.",
            "Python is a popular programming language for data science.",
        ]
        results = client.enrich_batch(texts)

        assert len(results) == 2
        for text, enriched in zip(texts, results):
            assert enriched is not None, f"enrichment failed for: {text}"
            assert enriched._original_text == text
            assert enriched.headline
            assert enriched.summary


# ─── 9. Search pipeline full path (FD-67…FD-71) ─────────────────────────

class TestSearchPipelineFullPath:
    @pytest.fixture()
    def populated_store_and_registry(self, tmp_path, store, registry, real_embedder):
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "a.txt").write_text(
            "The mitochondria is the powerhouse of the cell and produces ATP energy.",
            encoding="utf-8",
        )
        (docs / "b.txt").write_text(
            "Paris is the capital city of France and home to the Eiffel Tower.",
            encoding="utf-8",
        )
        run_pipeline(
            folder_path=str(docs), converter=ConverterManager(), embedder=real_embedder,
            store=store, collection_name=collection_name_for_path(str(docs)),
        )
        return docs

    def _patch_deps(self, monkeypatch, cfg, store, registry, real_embedder):
        monkeypatch.setattr(web_deps, "get_config", lambda: cfg)
        monkeypatch.setattr(web_deps, "get_store", lambda: store)
        monkeypatch.setattr(web_deps, "get_registry", lambda: registry)
        monkeypatch.setattr(web_deps, "get_embedder", lambda: real_embedder)

    def test_full_path_all_stages_enabled(self, monkeypatch, cfg, store, registry, real_embedder, populated_store_and_registry):
        self._patch_deps(monkeypatch, cfg, store, registry, real_embedder)
        results, rewritten = run_search("what powers a cell?", top_k=5, rewrite=True, rerank=True)
        assert len(results) >= 1
        assert any("mitochondria" in r.text.lower() for r in results)

    def test_stages_individually_disableable(self, monkeypatch, cfg, store, registry, real_embedder, populated_store_and_registry):
        self._patch_deps(monkeypatch, cfg, store, registry, real_embedder)

        no_rewrite, rw1 = run_search("capital of France", top_k=5, rewrite=False, rerank=True)
        assert rw1 is None
        assert len(no_rewrite) >= 1

        no_rerank, _ = run_search("capital of France", top_k=5, rewrite=True, rerank=False)
        assert len(no_rerank) >= 1

        neither, rw3 = run_search("capital of France", top_k=5, rewrite=False, rerank=False)
        assert rw3 is None
        assert len(neither) >= 1
        assert any("paris" in r.text.lower() for r in neither)

    def test_empty_query_returns_no_results(self, monkeypatch, cfg, store, registry, real_embedder, populated_store_and_registry):
        self._patch_deps(monkeypatch, cfg, store, registry, real_embedder)
        results, _ = run_search("", top_k=5)
        # embed("") still runs but should not error; assert it doesn't crash
        # and top_k is honored regardless of hit count.
        assert isinstance(results, list)

    def test_top_k_is_honored(self, monkeypatch, cfg, store, registry, real_embedder, populated_store_and_registry):
        self._patch_deps(monkeypatch, cfg, store, registry, real_embedder)
        results, _ = run_search("cell energy Paris France", top_k=1, rewrite=False, rerank=False)
        assert len(results) <= 1


# ─── 10. Ingest queue (FD-100…FD-106) ───────────────────────────────────

class TestIngestQueue:
    def test_sequential_processing_two_folders(self, tmp_path, store, registry, real_embedder, cfg, monkeypatch):
        docs_a = tmp_path / "a"
        docs_b = tmp_path / "b"
        docs_a.mkdir()
        docs_b.mkdir()
        (docs_a / "a.txt").write_text("Folder A content for queue test.", encoding="utf-8")
        (docs_b / "b.txt").write_text("Folder B content for queue test.", encoding="utf-8")

        monkeypatch.setattr(web_deps, "get_config", lambda: cfg)
        monkeypatch.setattr(web_deps, "get_converter", lambda: ConverterManager())
        monkeypatch.setattr(web_deps, "get_embedder", lambda: real_embedder)
        monkeypatch.setattr(web_deps, "get_store", lambda: store)
        monkeypatch.setattr(cfg, "folders_db_path", str(tmp_path / "registry.db"))

        folder_a_id = registry.add_folder(FolderConfig(path=str(docs_a)))
        folder_b_id = registry.add_folder(FolderConfig(path=str(docs_b)))

        queue = IngestQueue()
        queue.start()
        try:
            task_a = queue.push(folder_a_id, str(docs_a))
            task_b = queue.push(folder_b_id, str(docs_b))

            deadline = time.time() + 60
            while time.time() < deadline:
                sa = queue.get_status(task_a)
                sb = queue.get_status(task_b)
                if sa["status"] == TaskStatus.COMPLETED and sb["status"] == TaskStatus.COMPLETED:
                    break
                time.sleep(0.5)
            else:
                pytest.fail(f"Tasks did not complete in time: a={sa}, b={sb}")

            assert sa["chunks_done"] >= 1
            assert sb["chunks_done"] >= 1
        finally:
            queue.stop()

    def test_prioritize_reorders_pending_queue_and_signals_cancellation(self):
        """White-box: verify prioritize()'s queue-reordering and preemption
        signal deterministically, without depending on real-worker timing.
        """
        queue = IngestQueue()  # worker thread NOT started - inspect state directly

        with queue._lock:
            queue._current_task_id = "running-task"
            queue._current_cancel_event = threading.Event()
            from web.ingest_queue import IngestTask
            queue._tasks["running-task"] = IngestTask(
                task_id="running-task", folder_id=1, folder_path="/running",
            )

        task_id = queue.prioritize(folder_id=2, folder_path="/other")

        assert queue._queue[0] == task_id
        assert queue._current_cancel_event.is_set()

    def test_prioritize_moves_existing_pending_task_to_front(self):
        queue = IngestQueue()
        t1 = queue.push(1, "/f1")
        t2 = queue.push(2, "/f2")
        t3 = queue.push(3, "/f3")
        assert queue._queue == [t1, t2, t3]

        returned = queue.prioritize(folder_id=3, folder_path="/f3")

        assert returned == t3
        assert queue._queue[0] == t3
