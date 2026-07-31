# SRAG — Build Plan (V-Model)

**Audience:** the coding agent (Qwen 3.6) building this application from scratch on a new machine.
**Method:** V-model — design flows down the left arm (SD → DD → FD → code), verification flows up the right arm (unit → integration → system quality test).

---

## 1. Handover Package

You receive four documents. Each has a distinct role; do not treat them as interchangeable.

| Document | Role | Drives | Verified by |
|---|---|---|---|
| **SD.md** — System Design | System context, package architecture, data flows, threading model, persistence, API surface | **Phase 1** — project skeleton | System quality test |
| **DD.md** — Detail Design | Per-concern algorithm sequences, decision tables, schemas (**DD-01…DD-16**) | **Phase 2** — module logic | Integration test |
| **FD.md** — Function Design | Per-function contracts: signature, inputs, outputs, errors, state (**FD-01…FD-129** are in scope; **FD-130…FD-147** describe the out-of-scope `eval/` package — skip them) | **Phase 3** — source code | Unit test |
| **BUILD_PLAN.md** — this file | Build order, phase gates, test strategy, acceptance criteria | The whole build | — |

**Rule of precedence:** if the three design documents disagree, **FD wins for a function's contract**, **DD wins for an algorithm's sequence**, **SD wins for structure and cross-cutting concerns**. Record any conflict you find rather than silently choosing.

> **Note on cross-references:** SD/DD/FD contain occasional `ISSUE #nn` tags. These are traceability markers to the origin project's issue log and are **not** required reading — the requirement itself is always stated inline where the tag appears. Treat the tagged statement as normative.

---

## 2. Prerequisites

Verify all of these **before** Phase 1. Do not begin coding against a broken environment.

| Requirement | Check | Notes |
|---|---|---|
| Python ≥ 3.12 | `python --version` | |
| `uv` package manager | `uv --version` | All dependency and run commands use `uv`, not `pip`/`python` directly |
| Ollama running | `curl http://localhost:11434/api/version` | |
| Embedding model | `ollama pull qwen3-embedding` | 4096-dim; ~6.6 GB |
| LLM model | `ollama pull qwen3:4b` | Used for the query-rewrite stage |
| GPU available | `nvidia-smi` (or platform equivalent) | Target machine is GPU-backed. Confirm Ollama reports `size_vram > 0` via `/api/ps` after a first embed — if it is `0`, the model is running on CPU and bulk ingestion will be extremely slow |
| Disk space | ≥ 20 GB free | Models + vector store |

**Environment hygiene:** if `SSL_CERT_FILE`, `REQUESTS_CA_BUNDLE`, or `CURL_CA_BUNDLE` are set in the shell, confirm each points at a file that exists. A stale value breaks `ollama` and `tiktoken` at import time with errors that look unrelated to your code.

---

## 3. The V

```
  SD ─────────────────────────────────────────────────►  SYSTEM QUALITY TEST
  (structure, data flow,                                 (whole app: run → register →
   threading, persistence)                                ingest → search → MCP)
      │                                                        ▲
      ▼                                                        │
  DD ─────────────────────────────────────────────────►  INTEGRATION TEST
  (algorithms, sequences,                                 (module pairs + pipelines
   schemas, decision tables)                               against real Chroma/Ollama)
      │                                                        ▲
      ▼                                                        │
  FD ─────────────────────────────────────────────────►  UNIT TEST
  (function contracts:                                    (one function, fakes,
   in/out/errors/state)                                     tmp_path, no network)
      │                                                        ▲
      └──────────────────►  IMPLEMENTATION  ──────────────────┘
```

Each left-arm phase has a **gate**: you may not start the next phase until the current phase's exit criteria pass. Each right-arm level verifies exactly one left-arm document.

---

## 4. Phase 1 — Skeleton from SD

**Goal:** a runnable, empty-but-correct project structure. No business logic yet.

### 4.1 Tasks

1. `uv init`, then create `pyproject.toml` with these dependencies:

   ```toml
   [project]
   name = "srag"
   requires-python = ">=3.12"
   dependencies = [
       "anthropic>=0.116.0",
       "chromadb>=1.5.9",
       "fastapi>=0.139.0",
       # PINNED — see FD.md §6 Constants. Newer markitdown regresses PDF text
       # extraction on watermarked PDFs (emits reversed text). All four extras
       # are required or .docx/.pptx/.xlsx silently fail to convert.
       "markitdown[docx,pdf,pptx,xlsx]==0.1.5",
       "mcp[cli]>=1.28.1",
       "ollama>=0.6.2",
       "pdfminer-six==20251230",
       "pyyaml>=6.0.3",
       "tiktoken>=0.13.0",
       "uvicorn>=0.50.2",
   ]

   [dependency-groups]
   dev = ["httpx>=0.28.1", "pytest>=9.1.1", "reportlab>=5.0.0"]

   [tool.pytest.ini_options]
   pythonpath = ["."]
   markers = ["integration: requires external services (Ollama, ChromaDB)"]
   ```

2. Create the package tree exactly as specified in **SD.md §4**, with `__init__.py` in each package and every module present as a stub. Omit `eval/` — it is out of scope for this build.

3. Implement the three foundation modules fully (they have no dependencies and everything else imports them):
   - `config.py` — per **FD-01…FD-06** and **DD-01**. Dataclass tree, `${ENV}` expansion, `SRAG_*` env overrides, required-field validation, `_parse_bool`.
   - `log.py` — per **FD-07**. Logger named `srag`, stdout + file handlers, idempotent.
   - `models.py` — per **FD-08…FD-10**. `SearchResult`, `ChunkWithMetadata`, `generate_chunk_id`.

4. Create `config.yaml`:

   ```yaml
   chromadb_path: "${LOCALAPPDATA}/srag/chromadb"
   chromadb_tenant: "default_tenant"
   chromadb_database: "default_database"
   ollama_url: "http://localhost:11434"
   embedding_model: "qwen3-embedding"
   server:
     port: 9001
   mcp:
     transport: "stdio"
   search:
     top_k: 10
     rewrite_enabled: true
     rerank_enabled: true
     retrieval_k: 20
     final_k: 10
   chunking:
     max_tokens: 768
     overlap: 64
   llm:
     enrichment: "anthropic/claude-sonnet-5"
     rewrite: "ollama/qwen3:4b"
     rerank: "anthropic/claude-sonnet-5"
     eval: "anthropic/claude-sonnet-5"
   anthropic:
     base_url: ""
   enrichment:
     enabled: false
     workers: 1
   sync:
     interval_minutes: 30
     auto_start: true
   pipeline:
     embed_batch_size: 50
     max_retries: 3
   logging:
     level: "INFO"
     file: "./logs/srag.log"
   ```

   The `anthropic/*` stages activate only when `ANTHROPIC_API_KEY` is set; otherwise those stages log a warning and degrade gracefully (see FD-67). `enrichment.enabled: false` and `embed_batch_size: 50` are deliberate conservative defaults — tune only after the system quality test passes.

5. `run.bat` and `run.sh` per **FD-84/85**: verify `python`/`uv` present, warn if Ollama unreachable, then `uv run python -m web.app`.

### 4.2 Exit gate — Phase 1

- [ ] `uv sync` completes; `uv run python -c "import config, log, models"` succeeds
- [ ] `uv run python -c "from config import load_config; print(load_config())"` prints a fully populated `Config`
- [ ] Every module named in SD.md §4 exists (stub or real); `eval/` absent
- [ ] Unit tests for `config.py` pass (see §7.1)

---

## 5. Phase 2 — Modules from DD

**Goal:** each module implements its documented algorithm. Build strictly in dependency order — a module is only started once everything it imports is complete.

### 5.1 Build order

| # | Module | DD reference | Depends on |
|---|---|---|---|
| 1 | `store/chromadb_store.py` | **DD-02** | config, models |
| 2 | `embedding.py` | **DD-03** | config |
| 3 | `ingest/chunker.py` | **DD-04** | — |
| 4 | `ingest/converter.py` | — (see FD-21…FD-25) | — |
| 5 | `ingest/folder_registry.py` | **DD-05, DD-06** | config |
| 6 | `ingest/scanner.py` | **DD-07** | folder_registry |
| 7 | `ingest/retry.py` | **DD-12** | — |
| 8 | `llm/base.py`, `llm/factory.py`, `llm/ollama_provider.py`, `llm/anthropic_provider.py` | **DD-13, DD-14** | config |
| 9 | `ingest/enrichment.py` | **DD-10, DD-11** | llm |
| 10 | `web/progress.py` | — (FD-95…FD-99) | — |
| 11 | `ingest/pipeline.py` | **DD-08** | 1–10 |
| 12 | `ingest/sync.py` | **DD-09** | 1–11 |
| 13 | `search/rewriter.py`, `search/merge.py`, `search/reranker.py` | **DD-16** (merge dedup); FD-72…FD-77 | llm |
| 14 | `search/pipeline.py` | **DD-15**; FD-67…FD-71 | store, embedding, 13 |
| 15 | `mcp_server/tools.py`, `mcp_server/server.py` | — (FD-78…FD-83) | 14 |
| 16 | `web/dependencies.py` | — (FD-87…FD-94) | 1, 2, 5, 15 |
| 17 | `web/ingest_queue.py` | — (FD-100…FD-112) | 11, 12, 16 |
| 18 | `web/routes.py` | — (FD-113…FD-129; SD §10 API table) | 16, 17 |
| 19 | `web/app.py` | — (FD-84…FD-86) | 18 |
| 20 | `web/static/` (`index.html`, `css/styles.css`, `js/app.js`) | — (SD §10 Web UI table) | 18 |

**Numbering note:** `FD-116` does not exist — the number was retired along with a removed endpoint, so the route entries run FD-113…FD-115 then FD-117…FD-129. This is a gap, not a missing document.

### 5.2 Load-bearing design constraints

These are easy to implement "almost right" and hard to detect later. They are stated in the design docs; repeated here because getting them wrong is expensive.

| Constraint | Source | Why it matters |
|---|---|---|
| Create every collection with `metadata={"hnsw:space": "cosine"}` | DD-02, FD-16 | ChromaDB defaults to `l2`. Mixing metrics makes the cross-collection merge-sort meaningless. Cannot be changed after creation — the collection must be recreated. |
| Collection name = **bare** `sha256(normcase(abspath(path)))`, no prefix | DD-05, FD-29 | This datastore has a single owner; names must be deterministic across path spellings. |
| Route `.pdf` through the **CLI subprocess**; resolve the CLI from **this project's own venv** (`Path(sys.executable).parent`), then PATH | FD-24, FD-25 | The markitdown Python API mis-extracts text on some watermarked PDFs. Never hardcode a path into another application's install tree. |
| A conversion that returns `None`, or yields zero chunks, must be appended to `result.errors` **and** `progress.errors` — never skipped silently | DD-08 error table, FD-52, FD-54 | Otherwise a folder whose every file failed reports `status=ready, 0 chunks, 0 errors`, hiding total data loss. |
| SQLite connections use `check_same_thread=False`; background work gets its **own** `FolderRegistry` via `create_registry()` | FD-31, FD-94 | FastAPI runs sync handlers in a thread pool; the ingest queue and sync scheduler are separate threads. |
| Chunk IDs are deterministic (`generate_chunk_id`) so re-ingest **upserts** rather than duplicating | FD-10, DD-08 | Idempotency is what makes re-running ingest safe. |
| Every LLM-dependent stage degrades gracefully when its provider is unavailable | FD-67, FD-72, FD-75 | The app must remain fully functional with zero LLMs configured. |

### 5.3 Exit gate — Phase 2

- [ ] All 20 modules import cleanly; no stubs remain in scope
- [ ] `uv run python -m web.app` starts and serves `GET /api/status`
- [ ] Integration tests pass (see §7.2)

---

## 6. Phase 3 — Source detail from FD

**Goal:** every function matches its FD contract exactly. Phase 2 makes it work; Phase 3 makes it correct at the boundaries.

### 6.1 Method

Walk **FD-01 → FD-129** in order (stop there — FD-130…FD-147 belong to the out-of-scope `eval/` package). For each entry, verify the implementation against all five facets and fix any mismatch:

1. **Signature** — exact name, parameter names, defaults, return type
2. **Inputs** — types and accepted forms
3. **Outputs** — exact shape (e.g. `list[dict]` with keys `text`, `metadata`, `distance`, `collection`)
4. **Errors** — the documented exception type for each failure mode, no broader and no narrower
5. **State** — module-level caches, singletons, lazy initialisation

### 6.2 Boundary behaviours to get right

FD specifies these explicitly; they are the usual source of defects:

- `top_k=0` must mean zero, not "fall back to the default" (`if x is not None`, never `x or default`)
- Empty/whitespace query → return `[]` **before** calling Ollama or ChromaDB
- Empty document → zero chunks, not one empty chunk
- Reranker: handle out-of-range indices, duplicate indices, and missing indices (append the missing ones) so the output length always equals the input length
- `merge_results`: deduplicate by `source::chunk_index`, original-query results first
- Registering an already-registered folder → `ValueError`; a non-existent path → `FileNotFoundError`
- `unregister()` must cascade to the `files` and `file_errors` tables
- Context managers (`__enter__`/`__exit__`) on `ChromaStore` and `FolderRegistry`

### 6.3 Exit gate — Phase 3

- [ ] Every FD entry in scope is implemented and matches its contract
- [ ] Full unit suite passes (see §7.1)
- [ ] System quality test passes (see §7.3)

---

## 7. Verification (right arm of the V)

Each level verifies one document. A level may only be declared complete when the corresponding design document is fully covered.

### 7.1 Unit test — verifies FD

**Scope:** one function at a time. No network, no external services. Use `pytest` `tmp_path`, hand-written fakes (a `FakeEmbedder` / `FakeLLM` class, not a mocking framework), and `monkeypatch`.

**Rules**
- Mark nothing here `@pytest.mark.integration`; the whole suite must run offline via `uv run pytest tests/ -m "not integration"`
- Test the **error** path and the **boundary** values, not just the happy path — for every numeric parameter test `0` and `None`
- When monkeypatching, patch the name **where it is used**, not where it is defined
- Assert on specific values; a test that would still pass with the bug present is not a test

**Required coverage**

| Test file | Verifies |
|---|---|
| `test_config.py` | FD-01…FD-06 — YAML load, missing required field, defaults, env override, `${ENV}` expansion, missing file |
| `test_models.py` | FD-08…FD-10 — deterministic chunk IDs |
| `test_chromadb_store.py` | FD-14…FD-20 — create/upsert/query/delete against a `tmp_path` `PersistentClient`; **assert new collections report `hnsw:space == "cosine"`**; cross-collection merge ordering; missing-path error |
| `test_converter.py` | FD-21…FD-25 — `.txt`/`.md` read-as-is, metadata fields, unsupported extension → `None`, missing file → `FileNotFoundError`, CLI resolves inside this venv |
| `test_chunker.py` | FD-26…FD-28 — short text → 1 chunk, long text → many, overlap correctness, empty → `[]` |
| `test_folder_registry.py` | FD-29…FD-46 — register/duplicate/nonexistent, collection name is a bare 64-char SHA-256, unregister cascades to `files` + `file_errors`, file upsert, error log |
| `test_scanner.py` | FD-47…FD-50 — extension filter, recursion, new/modified/deleted detection |
| `test_retry.py` | FD-59 — retries then succeeds, exhausts and re-raises the original exception, respects the exception filter |
| `test_llm_factory.py` | FD-61 — routes `ollama/*` and `anthropic/*`, bad spec → `ValueError`, missing API key → `ValueError` |
| `test_enrichment.py` | FD-55…FD-58 — `as_text()` format, retry, **partial batch failure returns `None` for the failed item without discarding the successes** |
| `test_merge.py` | FD-73/74 — dedup, ordering, empty inputs |
| `test_reranker.py` | FD-75…FD-77 — valid order, out-of-range, duplicates, missing indices, LLM failure → original order |
| `test_rewriter.py` | FD-72 — rewrite, and fallback to the original query on failure |
| `test_search_pipeline.py` | FD-67…FD-71 — empty query, `top_k` honoured, rewrite disabled, rerank disabled, fallback when a stage fails |
| `test_pipeline.py` | FD-52 — bad file does not stop the run; **a file that fails conversion appears in `result.errors`** |
| `test_progress.py` | FD-95…FD-99 — `to_dict()` percent, division-by-zero safety |
| `test_api.py` | SD §10 — every endpoint's success and error status codes, using `create_app(use_lifespan=False)` with injected fakes |
| `test_mcp.py` | FD-82 — the tool is listed with the correct schema |

**Exit:** all unit tests green offline; every in-scope FD entry has at least one assertion.

### 7.2 Integration test — verifies DD

**Scope:** module pairs and whole pipelines against **real** ChromaDB (in `tmp_path`) and **real** Ollama. Mark every test `@pytest.mark.integration`; run with `uv run pytest tests/ -m integration`.

| Test | DD verified | Assertion |
|---|---|---|
| Embedder round-trip | DD-03 | Single and batch embed return 4096-dim vectors, count matches input |
| Cross-collection search | DD-02 | Results from multiple collections merge into one ascending-distance list; empty and corrupt collections are skipped without failing the search |
| Converter on a real PDF | FD-22 | Non-empty text; **zero reversed-text artifacts** (see §7.3 quality gate) |
| Converter on real `.docx`/`.pptx`/`.xlsx` | FD-22 | Each returns non-empty text — proves the markitdown extras are installed |
| Ingest a folder | DD-08 | Chunks land in ChromaDB; registry `doc_count`/`chunk_count`/`last_indexed` updated; status `ready` |
| Ingest idempotency | DD-08 | Running twice yields the same chunk count (upsert, not duplicate) |
| Ingest with a failing file | DD-08 | The failure is present in `result.errors`; other files still ingest |
| Sync — new / modified / deleted | DD-09 | New file adds chunks; modified file replaces them; deleted file removes them; no-change run is a no-op |
| Enrichment batch | DD-10, DD-11 | Parallel enrichment succeeds; `original_text` preserved verbatim |
| Search pipeline full path | FD-70 | rewrite → dual search → merge → rerank → `final_k`; each stage individually disable-able |
| Ingest queue | FD-100…FD-106 | Sequential processing; `prioritize()` preempts the running job, which is re-queued with status `paused` |

**Exit:** all integration tests green; every DD section (DD-01…DD-16) exercised.

### 7.3 System quality test — verifies SD

**Scope:** the assembled system, driven exactly as a user would. No test doubles. This is the final gate.

**7.3.1 Functional walkthrough**

1. `run.bat` (or `run.sh`) → server starts on port 9001; `GET /api/status` returns `chromadb_connected: true`
2. Open `http://localhost:9001/ui` → Dashboard, Local Folders, Search, Settings pages all render
3. Register a folder containing at least one **PDF**, one **DOCX**, and one **TXT** → auto-ingest is enqueued
4. Watch progress: `GET /api/folders/{path}/status` reports advancing `files_done`/`chunks_done`; the UI progress bar tracks it
5. On completion: `doc_count` and `chunk_count` are non-zero and **match the number of files actually converted**
6. Search a phrase you know appears in one of the documents → the correct chunk is returned, with sane `distance`, correct `filename`/`source` metadata
7. Toggle rewrite and rerank in Settings → `GET /api/config` reflects it and search still returns results with each stage off
8. Remove the folder → its collection is deleted and it disappears from `GET /api/folders`
9. Start the MCP server (`uv run mcp_server/server.py`), call `srag_search` over stdio → same results as the REST endpoint
10. Restart the server → registered folders and their chunk counts survive (persistence per SD §8)

**7.3.2 Quality gates** — the system may pass every functional step above and still be unfit. Check all four.

| Gate | How to measure | Pass condition |
|---|---|---|
| **Text fidelity** | Sample 20 stored chunks across file types and read them | Text is coherent prose/tables. **Zero** chunks containing reversed text (search a converted PDF's text for the reverse of a string you know is in it — e.g. reverse of `"disclosure"` → `erusolcsid`). Printable-character ratio > 0.95 |
| **Metadata integrity** | Fetch all chunks for one file | `chunk_index` contiguous `0..n-1`; `total_chunks` consistent; `filename`, `source`, `parent_document_id` populated on every chunk |
| **Retrieval sanity** | 5–10 questions whose answers you can locate by hand | The expected chunk appears in the top 10 for each |
| **Failure visibility** | Ingest a folder containing one deliberately corrupt/empty file | `GET /api/folders/{path}/errors` lists that file; the folder's counts reflect only the files that really succeeded |

**7.3.3 Non-functional checks**

| Property | Check |
|---|---|
| GPU actually in use | `GET http://localhost:11434/api/ps` reports `size_vram > 0` during ingest |
| Throughput baseline | Record chunks/second on a known folder — this is the reference for any later tuning |
| Concurrency | Trigger a search while an ingest is running; both complete without error or database-lock failures |
| Resilience | Stop Ollama mid-ingest → per-file errors are recorded and the process does not crash; restart Ollama and re-run → completes |
| Clean shutdown | Ctrl-C → queue and scheduler threads stop; no orphaned processes remain |

**Exit:** every functional step passes, all four quality gates pass, all non-functional checks pass.

---

## 8. Definition of Done

- [ ] Phases 1–3 gates all passed, in order
- [ ] Unit suite green offline: `uv run pytest tests/ -m "not integration"`
- [ ] Integration suite green: `uv run pytest tests/ -m integration`
- [ ] System quality test §7.3 fully passed, including all four quality gates
- [ ] `run.bat` starts the app from a clean checkout after `uv sync`
- [ ] No module contains a stub or `TODO` within the agreed scope
- [ ] Deviations from SD/DD/FD are documented with rationale

---

## 9. Scope Boundary

**In scope:** `config.py`, `log.py`, `models.py`, `embedding.py`, `store/`, `ingest/`, `llm/`, `search/`, `mcp_server/`, `web/` (incl. static UI), `run.bat`/`run.sh`, `tests/`.

**Out of scope:** the `eval/` package (retrieval metrics, LLM-as-judge, configuration comparison) — specified in **FD-130…FD-147** and reachable via the `POST /api/eval/*` endpoints (**FD-130…FD-132**). It *measures* retrieval quality but does not *affect* it, and its value depends on a hand-curated question set matched to the target document corpus. Omit those three endpoints from `web/routes.py`. It can be added later without touching any module built here — it consumes `search/pipeline.py` through the same public interface as the REST API.

---

## 10. Reporting

On completion, produce a short build report containing:

1. Phase gate results, with dates
2. Unit / integration / system test results (pass counts, and any test skipped with the reason)
3. The four quality-gate measurements from §7.3.2, with the actual numbers
4. Throughput baseline (chunks/second) and confirmation that the GPU was in use
5. Any deviation from SD/DD/FD, and why
6. Any defect found in the design documents themselves
