# SRAG — Usage Guide

## Prerequisites

- **Ollama** installed and running, with an embedding model pulled
- **Python 3.12+** and **[uv](https://docs.astral.sh/uv/)** — `install.bat`/`install.sh` will install `uv` for you if it's missing; `uv` in turn manages the correct Python version automatically per `pyproject.toml`

SRAG is standalone — it creates its own ChromaDB on first run (`%LOCALAPPDATA%\srag\chromadb` on Windows, `~/Library/Application Support/srag/chromadb` on macOS, `$XDG_DATA_HOME/srag/chromadb` or `~/.local/share/srag/chromadb` on Linux). No other application is required.

### Verify Ollama

```bash
curl http://localhost:11434/api/version
ollama list
```

If the embedding model configured in `config.yaml` (`embedding_model`, default `qwen3-embedding-8b`) isn't listed, pull it:

```bash
ollama pull qwen3-embedding:8b
ollama cp qwen3-embedding:8b qwen3-embedding-8b   # alias to match config.yaml exactly
```

`install.bat`/`install.sh` do this automatically. The other models `config.yaml` references (`llm.rewrite`/`llm.rerank`/`llm.enrichment`/`llm.eval`) are **optional** — SRAG works fully without them; those specific stages just become a no-op if the model isn't available.

## Install

One-time setup — installs `uv` if missing, runs `uv sync`, checks Ollama and pulls the embedding model, and (best-effort) registers the MCP server with Claude Code if the `claude` CLI is present:

```bash
install.bat        # Windows
./install.sh        # macOS/Linux
```

## Quick Start

### Option 1: Web UI (recommended)

```bash
run.bat             # Windows
./run.sh              # macOS/Linux
```

Open http://localhost:9000 in your browser. You'll be redirected to the dashboard at `/static/index.html`.

### Option 2: Manual start

```bash
uv run python -m uvicorn web.app:create_app --factory --host 0.0.0.0 --port 9000
```

## Web UI

The web interface is available at `http://localhost:9000` with four pages:

### Dashboard

Overview of the system: registered folder count, total chunks, ChromaDB connection status, and a collections table (chunk count per collection).

### Local Folders

Manage document folders for ingestion:
- **Register** — enter a folder path; ingestion is enqueued automatically
- **Ingest / Sync** — re-run ingestion, or run an incremental sync (new/modified/deleted files)
- **Prioritize** — preempt the ingest queue so this folder runs next (the currently-running job, if any, pauses and re-queues automatically — safe, since re-ingest is idempotent)
- **Remove** — unregister a folder and delete its ChromaDB collection
- **Toggle excluded / single-search** — exclude a folder from search, or restrict search to just that one folder

Supported file formats: PDF, DOCX, PPTX, XLSX, TXT, MD, EPUB, HTML, PNG/JPG (image OCR requires the proprietary ABBYY FineReader SDK, which isn't bundled — image files return a clear error without it, everything else works fully).

### Search

Search across all registered (non-excluded) folders' collections:
- Enter a query and search; adjust `top_k` (default 10)
- Toggle query rewrite (LLM-expanded cross-lingual EN/VI variants) and reranking (LLM relevance re-scoring) independently
- Results show filename, source, chunk index, text excerpt, distance, and (if reranked) rerank score

### Settings

View and edit runtime configuration: ChromaDB path, Ollama URL, embedding model, server port, chunking parameters, and the rewrite/rerank toggles. Changes via the UI use `PATCH /api/config` — they take effect immediately but are **not** persisted to `config.yaml` (restart reverts to the file's values).

## REST API

All endpoints are under `http://localhost:9000/api/` (16 endpoints):

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/status` | Folder count, total chunks, collection count, ChromaDB connection status |
| GET | `/api/folders` | List all registered folders with status and live task info |
| POST | `/api/folders` | Register a folder `{"path": "..."}` (auto-ingests) |
| DELETE | `/api/folders/{path}` | Unregister a folder and delete its collection |
| POST | `/api/folders/{path}/ingest` | Queue a re-ingest |
| POST | `/api/folders/{path}/prioritize` | Preempt the queue to ingest this folder next |
| POST | `/api/folders/{path}/sync` | Queue an incremental sync (new/modified/deleted files) |
| GET | `/api/folders/{path}/status` | Folder detail + live ingest task progress |
| GET | `/api/folders/{path}/errors` | Per-file conversion/ingest errors for this folder |
| GET | `/api/folders/{path}/files` | File listing (size, modified time) |
| PATCH | `/api/folders/{path}/toggle-excluded` | Toggle exclude-from-search |
| PATCH | `/api/folders/{path}/toggle-single` | Toggle single-folder ("solo") search mode |
| POST | `/api/search` | Search `{"query": "...", "top_k": 10, "rewrite_query": false, "rerank_results": false}` |
| GET | `/api/collections` | List all ChromaDB collections with chunk counts |
| GET | `/api/config` | Current runtime configuration |
| PATCH | `/api/config` | Update runtime configuration (in-memory only, not persisted) |

`GET /` and `GET /ui` redirect to `/static/index.html`.

### Example: Search via curl

```bash
curl -X POST http://localhost:9000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "software architecture", "top_k": 5, "rewrite_query": true, "rerank_results": true}'
```

### Example: Register a folder

```bash
curl -X POST http://localhost:9000/api/folders \
  -H "Content-Type: application/json" \
  -d '{"path": "C:\\Data\\my_documents"}'
```

Registering auto-enqueues ingestion — poll `GET /api/folders/{url-encoded-path}/status` for progress.

## Ingest Pipeline (CLI)

Ingest a folder directly, without the web server:

```bash
uv run python -m ingest.pipeline "C:\Data\my_documents"
```

Sequence: scan folder → convert files (via the `markitdown` CLI, resolved from this project's own venv) → chunk text (768 tokens, 64 overlap by default) → embed via Ollama → upsert into ChromaDB. Chunk IDs are deterministic (SHA-256 of source path + index + content), so re-running is idempotent — it upserts rather than duplicating.

## MCP Server

The MCP server exposes SRAG's search and ingest functionality over stdio for AI assistants like Claude Code.

### Start standalone

```bash
uv run python -m mcp_server.server
```

### Register with Claude Code

`install.bat`/`install.sh` do this automatically if the `claude` CLI is present. To do it manually:

```bash
claude mcp add srag --scope user -- uv run --directory "<path-to-srag>" python -m mcp_server.server
```

`--directory` is required — `claude mcp add` does not preserve a working-directory setting on its own, so the server needs an explicit path to find its own package regardless of where Claude Code happens to launch it from.

### Tools

| Tool | Parameters | Description |
|------|------------|--------------|
| `srag_search` | `query` (str, required), `top_k` (int, default 10), `rewrite_query` (bool, default false), `rerank_results` (bool, default false) | Search the knowledge base; returns ranked chunks |
| `srag_ingest_file` | `file_path` (str, required) | Queue the file's containing folder for ingestion |
| `srag_ingest_folder` | `folder_path` (str, required) | Queue a folder for ingestion |
| `srag_stats` | — | Document/chunk/collection counts |
| `srag_health` | — | Storage/embedding/LLM connectivity status |

## Configuration

All settings are in `config.yaml`:

| Setting | Default | Description |
|---------|---------|-------------|
| `chromadb_path` | `${LOCALAPPDATA}/srag/chromadb` | Path to SRAG's own ChromaDB (auto-created; resolves correctly on macOS/Linux) |
| `chromadb_tenant` | `default_tenant` | ChromaDB tenant name |
| `chromadb_database` | `default_database` | ChromaDB database name |
| `ollama_url` | `http://localhost:11434` | Ollama server URL |
| `embedding_model` | `qwen3-embedding-8b` | Embedding model name |
| `server.port` | `9000` | Web server port |
| `search.top_k` | `10` | Default number of search results |
| `search.rewrite_enabled` | `true` | Enable LLM query rewriting by default |
| `search.rerank_enabled` | `true` | Enable LLM reranking by default |
| `search.retrieval_k` | `20` | Candidates retrieved per query variant before merge/rerank |
| `search.final_k` | `10` | Results returned after merge/rerank |
| `chunking.max_tokens` | `768` | Max tokens per chunk |
| `chunking.overlap` | `64` | Token overlap between chunks |
| `llm.rewrite` | `ollama/qwen3-5-9b` | Model for query rewriting (`provider/model`) |
| `llm.rerank` | `ollama/qwen3-5-9b` | Model for reranking search results |
| `llm.enrichment` | `ollama/qwen3-5-9b` | Model for per-chunk headline + summary (only used if `enrichment.enabled: true`) |
| `llm.eval` | `ollama/qwen3-5-9b` | Reserved for future evaluation tooling |
| `anthropic.base_url` | `""` (api.anthropic.com) | Set for a self-hosted/proxied Anthropic endpoint |
| `enrichment.enabled` | `false` | Whether ingestion enriches chunks with LLM-generated metadata |
| `enrichment.workers` | `1` | Concurrent enrichment workers |
| `sync.interval_minutes` | `30` | Background sync interval (if a scheduler consumes this — see note below) |
| `sync.auto_start` | `true` | — |
| `pipeline.embed_batch_size` | `50` | Chunks per embedding batch during ingest |
| `pipeline.max_retries` | `3` | Retry count for transient failures |
| `logging.level` | `INFO` | Log level (DEBUG, INFO, WARNING, ERROR) |
| `logging.file` | `./logs/srag.log` | JSON-lines log file (console output is also always on, human-readable) |

Any `llm.*` stage pointed at `ollama/...` degrades gracefully (logs a warning, skips that stage) if the model isn't pulled — search itself never depends on these, only rewrite/rerank/enrichment do. `anthropic/...` specs require `ANTHROPIC_API_KEY` to be set.

> **Note**: `sync.*` config exists and is read, but nothing in the live web app currently runs a background scheduler off it — `ingest/sync.py`'s `full_sync`/`incremental_sync`/`smart_sync` are fully implemented and integration-tested, but the web UI's "Sync" button and the ingest queue call the ingest pipeline directly rather than going through a scheduled loop.

Environment variable overrides (take precedence over `config.yaml`):

| Env Variable | Overrides |
|--------------|-----------|
| `SRAG_CHROMADB_PATH` | `chromadb_path` |
| `SRAG_CHROMADB_TENANT` | `chromadb_tenant` |
| `SRAG_CHROMADB_DATABASE` | `chromadb_database` |
| `SRAG_OLLAMA_URL` | `ollama_url` |
| `SRAG_EMBEDDING_MODEL` | `embedding_model` |
| `SRAG_EMBEDDING_DIM` | `embedding_dim` |
| `SRAG_LLM_PROVIDER` | `llm_provider` |
| `SRAG_ANTRHOPIC_API_KEY` *(sic)* | `anthropic_api_key` |
| `SRAG_OllAMA_TIMEOUT` *(sic)* | `ollama_timeout` |
| `SRAG_TOP_K` | `top_k` |
| `SRAG_LOG_LEVEL` | `logging.level` |
| `SRAG_SERVER_PORT` | `server.port` |

(The two `*(sic)*` entries have inconsistent capitalization in the actual code — not a typo in this doc, matching what `config.py` actually checks.)

## Running Tests

```bash
# Unit tests (no external services needed) - 246 tests
uv run pytest tests/ -m "not integration"

# Integration tests (requires a running Ollama with the configured models pulled) - 26 tests
uv run pytest tests/ -m integration

# All tests - 272 total
uv run pytest tests/
```

## Project Structure

```
srag/
├── config.yaml                # all settings
├── config.py                  # config loader, ${VAR} expansion, validation
├── models.py                  # SearchResult, ChunkWithMetadata, generate_chunk_id
├── log.py                     # logging setup (console + JSON-lines file)
├── embedding.py                # Ollama embedding client wrapper
├── install.bat / install.sh   # one-time environment setup (uv, deps, Ollama model, MCP)
├── run.bat / run.sh            # start the web UI or MCP server
├── store/
│   └── chromadb_store.py      # ChromaDB wrapper, multi-collection search + merge
├── ingest/
│   ├── converter.py           # markitdown CLI subprocess conversion (PDF/DOCX/PPTX/XLSX/...)
│   ├── chunker.py             # token-based text chunking
│   ├── scanner.py             # file discovery, standalone or registry-backed
│   ├── folder_registry.py     # SQLite folder + file tracking
│   ├── sync.py                # full/incremental/smart sync (new/modified/deleted)
│   ├── retry.py               # exponential-backoff retry decorator
│   └── pipeline.py            # scan → convert → chunk → (enrich) → embed → upsert
├── llm/
│   ├── base.py                # LLMProvider protocol, GenerationResult
│   ├── factory.py             # create_client("provider/model") / create_llm_provider
│   ├── ollama_provider.py     # Ollama chat API provider
│   ├── anthropic_provider.py  # Anthropic SDK provider
│   └── enricher.py            # per-chunk LLM enrichment (headline/summary/keywords/...)
├── search/
│   ├── rewriter.py            # query rewriting with graceful fallback
│   ├── merge.py                # result dedup + distance averaging across query variants
│   └── reranker.py            # LLM-based result reranking
├── mcp_server/
│   ├── server.py               # MCP stdio server, tool registration
│   └── tools.py                # SragTools — search/ingest/stats/health implementations
├── web/
│   ├── app.py                  # FastAPI factory + uvicorn entry point
│   ├── routes.py               # REST API endpoints (16)
│   ├── dependencies.py         # shared singletons, lifespan, run_search()
│   ├── ingest_queue.py         # background worker: sequential ingest with prioritize()
│   └── static/                 # SPA (index.html, style.css, app.js)
└── tests/                      # 246 unit + 26 integration tests
```
