# System Design — SRAG

## 1. System Overview

SRAG is a **standalone** Python-based RAG (Retrieval Augmented Generation) system. It owns its own ChromaDB datastore and has no runtime dependency on any other application. It provides document ingestion with LLM enrichment, semantic search with query rewriting and reranking, a web dashboard, an MCP tool server, and a retrieval evaluation framework.

## 2. System Context

```
┌──────────────────────────────────────────────────────────┐
│                      User / AI Assistant                 │
│  (Browser at :9001/ui)    (MCP client via stdio)         │
└──────┬──────────────────────────────┬────────────────────┘
       │ HTTP                         │ MCP (stdio)
       ▼                              ▼
┌─────────────┐              ┌─────────────────┐
│  FastAPI     │              │  MCP Server     │
│  Web Server  │              │  (FastMCP)      │
│  port 9001   │              │                 │
└──────┬───────┘              └────────┬────────┘
       │                               │
       ▼                               ▼
┌──────────────────────────────────────────────────────────┐
│                   Core Services                          │
│  SearchPipeline · IngestPipeline · SyncPipeline           │
│  IngestQueue · SyncScheduler · FolderRegistry             │
└──────┬──────────────┬──────────────────┬─────────────────┘
       │              │                  │
       ▼              ▼                  ▼
┌───────────┐  ┌────────────┐  ┌──────────────────┐
│ ChromaDB  │  │ Ollama     │  │ SQLite           │
│ (own DB,  │  │ :11434     │  │ folders.db       │
│ sole      │  │ embed+LLM  │  │ %LOCALAPPDATA%/  │
│ owner)    │  └────────────┘  │ srag/        │
│ %LOCAL…/  │                  └──────────────────┘
│ srag/ │
│ chromadb  │
└───────────┘
```

**Standalone boundary:** every dependency above is either owned by srag (ChromaDB, SQLite, logs) or a generic local service (Ollama). There is no runtime dependency on the separate SRAG (Java) application — not on its datastore, its collections, or its install directory.

## 3. External Dependencies

| Dependency | Purpose | Interface |
|---|---|---|
| ChromaDB (persistent) | Vector storage, owned solely by srag | PersistentClient, file-based at `%LOCALAPPDATA%/srag/chromadb` (default tenant/database) |
| Ollama | Embedding (`qwen3-embedding`) + LLM (`qwen3:4b`) | HTTP API at `:11434` |
| Anthropic API | Optional LLM provider (eval judge, enrichment) | HTTP API via `anthropic` SDK |
| SQLite | Folder/file tracking, error logs | File-based at `%LOCALAPPDATA%/srag/folders.db` |
| Markitdown | Document conversion (PDF, DOCX, PPTX, XLSX, etc.) | **Pinned** `markitdown[docx,pdf,pptx,xlsx]==0.1.5` + `pdfminer-six==20251230` — all four extras required (ISSUE #36); 0.1.6 regresses PDF text (ISSUE #39). PDF via **CLI subprocess** resolved from srag's own venv (ISSUE #37); other formats via the Python API |

## 4. Package Architecture

```
SRAG/
├── config.py              # Configuration management
├── log.py                 # Logging setup
├── models.py              # Shared data models
├── embedding.py           # Ollama embedding client
├── store/                 # Data access layer
│   └── chromadb_store.py  #   ChromaDB operations
├── ingest/                # Document ingestion
│   ├── converter.py       #   File → Markdown conversion
│   ├── chunker.py         #   Text → token-based chunks
│   ├── enrichment.py      #   LLM headline + summary per chunk
│   ├── pipeline.py        #   Full ingest orchestration
│   ├── sync.py            #   Incremental sync (new/mod/del)
│   ├── scanner.py         #   Filesystem scanning + change detection
│   ├── folder_registry.py #   SQLite folder/file/error tracking
│   ├── retry.py           #   Exponential backoff decorator
│   └── scheduler.py       #   (legacy, replaced by ingest_queue)
├── llm/                   # LLM provider abstraction
│   ├── base.py            #   LLMProvider protocol
│   ├── factory.py         #   Provider factory ("provider/model" DSL)
│   ├── ollama_provider.py #   Ollama chat completion
│   └── anthropic_provider.py  # Anthropic Messages API
├── search/                # Search pipeline
│   ├── pipeline.py        #   Rewrite → dual search → merge → rerank
│   ├── rewriter.py        #   LLM query rewriting
│   ├── merge.py           #   Result deduplication + merge
│   └── reranker.py        #   LLM-based reranking
├── mcp_server/            # MCP tool server
│   ├── server.py          #   FastMCP server + tool registration
│   └── tools.py           #   SearchService facade
├── web/                   # Web server + UI
│   ├── app.py             #   FastAPI app factory + lifespan
│   ├── routes.py          #   19 REST API endpoints
│   ├── dependencies.py    #   Singleton DI container
│   ├── progress.py        #   Thread-safe ingest progress
│   ├── ingest_queue.py    #   Priority queue + sync scheduler
│   └── static/            #   SPA (HTML/CSS/JS)
├── eval/                  # Retrieval evaluation
│   ├── metrics.py         #   MRR, nDCG, keyword coverage
│   ├── loader.py          #   JSONL test dataset parser
│   ├── judge.py           #   LLM-as-judge scoring
│   ├── runner.py          #   Evaluation runner
│   ├── compare.py         #   3-config comparison
│   └── test_dataset.jsonl #   35 curated test questions
└── tests/                 # Unit + integration tests (111 unit)
```

## 5. Data Flow — Ingest

```
Folder on disk
  │
  ▼ scan_folder()
List of supported files (.pdf, .docx, .md, .txt, ...)
  │
  ▼ convert_file()          [Markitdown: CLI for .pdf, Python API otherwise, read-as-is for .txt/.md]
Markdown text + metadata      (failure here is recorded in result.errors, never skipped silently — ISSUE #38)
  │
  ▼ chunk_text()            [tiktoken, 768 tokens, 64 overlap per config.yaml]
List of text chunks
  │
  ▼ enrich_batch()          [optional, LLM: headline + summary]
Enriched chunks (headline + summary + original)
  │
  ▼ embedder.embed_batch()  [qwen3-embedding via Ollama]
Embedding vectors (4096-dim)
  │
  ▼ store.upsert()          [ChromaDB collection]
Stored in srag's own ChromaDB (default tenant/database)
```

## 6. Data Flow — Search

```
User query
  │
  ▼ rewrite_query()         [optional, LLM rewrites for search optimization]
Original query + rewritten query
  │
  ▼ embedder.embed()        [embed both queries]
  ▼ store.search()          [vector search across all collections]
Original results + rewritten results
  │
  ▼ merge_results()         [dedup by source+chunk_index, original first]
Merged candidates (~40)
  │
  ▼ rerank()                [optional, LLM orders by relevance]
Reranked results
  │
  ▼ [:final_k]             [top-K truncation, default 10]
Final results → API response
```

## 7. Threading Model

| Thread | Component | Purpose |
|---|---|---|
| Main (uvicorn) | FastAPI request handlers | HTTP request processing |
| IngestQueue worker | `IngestQueue._worker` | Sequential folder ingestion with preemption |
| SyncScheduler | `SyncScheduler._run` | Periodic change detection, enqueues sync jobs |
| Enrichment pool | `ThreadPoolExecutor(workers=3)` | Parallel LLM enrichment calls per file |

## 8. Persistence

| Store | Location | Contents |
|---|---|---|
| ChromaDB | `%LOCALAPPDATA%/srag/chromadb` | Document chunks + embeddings (owned solely by srag) |
| SQLite | `%LOCALAPPDATA%/srag/folders.db` | Registered folders, file tracking, error logs |
| config.yaml | Project root | All configuration (not modified at runtime) |
| Log file | `./logs/srag.log` | Application logs |

## 9. Configuration Hierarchy

```
config.yaml (base values)
  ↓ overridden by
${ENV_VAR} expansion in YAML values
  ↓ overridden by
SRAG_* environment variables (7 mappings)
  ↓ overridden by
PATCH /api/config (runtime-only, not persisted)
```

## 10. API Surface

### REST API (19 endpoints, prefix `/api`)

| Method | Path | Purpose |
|---|---|---|
| GET | `/status` | System status |
| GET | `/folders` | List registered folders |
| POST | `/folders` | Register + auto-ingest |
| DELETE | `/folders/{path}` | Unregister + delete collection |
| POST | `/folders/{path}/ingest` | Queue re-ingest |
| POST | `/folders/{path}/prioritize` | Priority preemption |
| POST | `/folders/{path}/sync` | Queue incremental sync |
| GET | `/folders/{path}/status` | Folder detail + progress |
| GET | `/folders/{path}/errors` | Error log |
| GET | `/folders/{path}/files` | File listing |
| PATCH | `/folders/{path}/toggle-excluded` | Toggle exclude flag |
| PATCH | `/folders/{path}/toggle-single` | Toggle single-search mode |
| POST | `/search` | Semantic search |
| GET | `/collections` | ChromaDB collections |
| GET | `/config` | Current configuration |
| PATCH | `/config` | Runtime config update |
| POST | `/eval/run` | Run evaluation |
| POST | `/eval/compare` | Start comparison (async) |
| GET | `/eval/compare` | Poll comparison result |

### MCP Tool

| Tool | Parameters | Purpose |
|---|---|---|
| `srag_search` | `query: str, top_k: int = 10` | Search via MCP (stdio transport) |

### Web UI (5 pages)

| Page | Features |
|---|---|
| Dashboard | Stat cards, collections table |
| Local Folders | Register, ingest/prioritize/remove, S/E/P toggles, progress bar, file detail modal |
| Search | Semantic search with relevance bars, pipeline indicators |
| Evaluation | Run eval, run comparison, metric tables with color coding |
| Settings | Config display, toggle switches, editable top_k |
