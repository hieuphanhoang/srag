# SRAG — Usage Guide

## Prerequisites

- **Python 3.12+** with [uv](https://docs.astral.sh/uv/) installed
- **Ollama** running locally with the `qwen3-embedding` model pulled

srag is standalone — it creates its own ChromaDB at `%LOCALAPPDATA%\srag\chromadb`
on first run. No other application is required.

### Verify prerequisites

```bash
# Check Ollama is running and model is available
curl http://localhost:11434/api/tags
```

If `qwen3-embedding` is not listed, pull it:

```bash
ollama pull qwen3-embedding
```

## Install

```bash
cd c:\00_Users\SRAG
uv sync
```

## Quick Start

### Option 1: Web UI (recommended)

```bash
run.bat
```

Open http://localhost:9001 in your browser. You'll be redirected to the dashboard at `/ui`.

### Option 2: Manual start

```bash
uv run python -m web.app
```

## Web UI

The web interface is available at `http://localhost:9001/ui` with five pages:

### Dashboard

Shows an overview of the system:
- **Folders** — number of registered folders
- **Total Chunks** — total chunks across all registered folders
- **Collections** — number of ChromaDB collections (one per registered folder)
- **ChromaDB** — connection status
- **Collections table** — all collections with chunk counts

### Local Folders

Manage document folders for ingestion:
- **Register** — enter a folder path and click Register
- **Ingest** — click Ingest to convert, chunk, embed, and store all supported files
- **Remove** — unregister a folder and delete its ChromaDB collection
- **Status badges** — `registered` (gray), `indexing` (blue pulse), `ready` (green), `error` (red)

Supported file formats: PDF, DOCX, PPTX, XLSX, CSV, TXT, MD, HTML, C, H

### Search

Search across all ingested collections:
- Enter a query and click Search (or press Enter)
- Adjust the top-K parameter (default 10)
- Results show filename, collection, text excerpt, and distance score

### Evaluation

Measure retrieval quality:
- **Run Eval** — score the current pipeline against the curated test dataset (MRR, nDCG, keyword coverage)
- **Run Comparison** — compare configurations (e.g. rewrite/rerank on vs off) side by side, with color-coded metric tables

### Settings

Displays current configuration: ChromaDB path, Ollama URL, embedding model, server port, chunking parameters.

## REST API

All endpoints are available at `http://localhost:9001/api/` (19 endpoints):

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/status` | Overall status (folder count, chunks, ChromaDB connection) |
| GET | `/api/folders` | List all registered folders with status |
| POST | `/api/folders` | Register a folder `{"path": "C:\\..."}` (auto-ingests) |
| DELETE | `/api/folders/{path}` | Unregister folder and delete its collection |
| POST | `/api/folders/{path}/ingest` | Queue re-ingest |
| POST | `/api/folders/{path}/prioritize` | Preempt the queue to ingest this folder next |
| POST | `/api/folders/{path}/sync` | Queue incremental sync (new/modified/deleted files) |
| GET | `/api/folders/{path}/status` | Folder detail + live ingest progress |
| GET | `/api/folders/{path}/errors` | Conversion/ingest error log for this folder |
| GET | `/api/folders/{path}/files` | File listing (size, type, modified) |
| PATCH | `/api/folders/{path}/toggle-excluded` | Toggle exclude-from-search flag |
| PATCH | `/api/folders/{path}/toggle-single` | Toggle single-folder search mode |
| POST | `/api/search` | Search `{"query": "...", "top_k": 10}` |
| GET | `/api/collections` | List all ChromaDB collections |
| GET | `/api/config` | Current configuration values |
| PATCH | `/api/config` | Runtime config update (not persisted to `config.yaml`) |
| POST | `/api/eval/run` | Run evaluation against the test dataset |
| POST | `/api/eval/compare` | Start a multi-config comparison (async) |
| GET | `/api/eval/compare` | Poll comparison result |

### Example: Search via curl

```bash
curl -X POST http://localhost:9001/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "software architecture", "top_k": 5}'
```

### Example: Register and ingest a folder

```bash
# Register
curl -X POST http://localhost:9001/api/folders \
  -H "Content-Type: application/json" \
  -d '{"path": "C:\\Data\\my_documents"}'

# Trigger ingestion
curl -X POST http://localhost:9001/api/folders/C%3A%5CData%5Cmy_documents/ingest
```

## Ingest Pipeline (CLI)

Ingest a folder directly from the command line without the web server:

```bash
uv run python -m ingest.pipeline "C:\Data\my_documents"
```

The pipeline: scan folder → convert files to Markdown (via `markitdown` CLI) → chunk text (768 tokens, 64 overlap) → embed via Ollama → upsert into ChromaDB.

Ingestion is idempotent — re-running produces the same chunks (deterministic IDs via SHA-256).

## MCP Server

The MCP server exposes a `srag_search` tool over stdio transport for AI assistants.

### Start standalone

```bash
uv run mcp_server/server.py
```

### Add to Claude Code

Add to `.claude/settings.json` or `.claude/settings.local.json`:

```json
{
  "mcpServers": {
    "srag": {
      "command": "uv",
      "args": ["run", "mcp_server/server.py"],
      "cwd": "c:/00_Users/SRAG"
    }
  }
}
```

### MCP tool: `srag_search`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | (required) | Search query |
| `top_k` | integer | 10 | Number of results to return |

Returns a ranked list of chunks with `text`, `metadata`, `distance`, and `collection`.

## Configuration

All settings are in `config.yaml`:

| Setting | Default | Description |
|---------|---------|-------------|
| `chromadb_path` | `${LOCALAPPDATA}/srag/chromadb` | Path to srag's own ChromaDB (auto-created on first run) |
| `chromadb_tenant` | `default_tenant` | ChromaDB tenant name |
| `chromadb_database` | `default_database` | ChromaDB database name |
| `ollama_url` | `http://localhost:11434` | Ollama server URL |
| `embedding_model` | `qwen3-embedding` | Embedding model name |
| `server.port` | `9001` | Web server port |
| `search.top_k` | `10` | Default number of search results |
| `chunking.max_tokens` | `768` | Max tokens per chunk |
| `chunking.overlap` | `64` | Token overlap between chunks |
| `llm.enrichment` | `anthropic/claude-sonnet-5` | Model for per-chunk headline + summary (`provider/model` DSL) |
| `llm.rewrite` | `ollama/qwen3:4b` | Model for query rewriting (runs per search, kept on local Ollama) |
| `llm.rerank` | `anthropic/claude-sonnet-5` | Model for reranking search results |
| `llm.eval` | `anthropic/claude-sonnet-5` | Model used as judge in evaluation |
| `anthropic.base_url` | `""` (api.anthropic.com) | Set for Foundry/Bedrock/self-hosted Anthropic endpoints |
| `logging.level` | `INFO` | Log level (DEBUG, INFO, WARNING, ERROR) |
| `logging.file` | `./logs/srag.log` | Log file path |

Frontier (`anthropic/...`) stages require `ANTHROPIC_API_KEY` (or `ANTHROPIC_FOUNDRY_API_KEY` +
`ANTHROPIC_FOUNDRY_RESOURCE`) to be set; without it, `create_llm()` logs a warning and that
stage falls back gracefully (e.g. rerank/enrichment simply skip) rather than failing the request.

Environment variable overrides (take precedence over YAML):

| Env Variable | Overrides |
|--------------|-----------|
| `SRAG_CHROMADB_PATH` | `chromadb_path` |
| `SRAG_CHROMADB_TENANT` | `chromadb_tenant` |
| `SRAG_CHROMADB_DATABASE` | `chromadb_database` |
| `SRAG_OLLAMA_URL` | `ollama_url` |
| `SRAG_EMBEDDING_MODEL` | `embedding_model` |
| `SRAG_LOG_LEVEL` | `logging.level` |
| `SRAG_SERVER_PORT` | `server.port` |

## Running Tests

```bash
# Unit tests (no external services needed)
uv run pytest tests/ -m "not integration"

# Integration tests (requires Ollama; uses srag's own ChromaDB)
uv run pytest tests/ -m integration

# All tests
uv run pytest tests/ -v
```

## Project Structure

```
SRAG/
├── config.yaml                # all settings
├── config.py                  # config loader + validation
├── models.py                  # SearchResult, ChunkWithMetadata
├── log.py                     # logging setup
├── embedding.py               # Ollama qwen3-embedding wrapper
├── run.bat / run.sh           # one-click startup scripts
├── store/
│   └── chromadb_store.py      # own ChromaDB (sole owner, default tenant/database)
├── ingest/
│   ├── converter.py           # markitdown CLI subprocess conversion
│   ├── chunker.py             # tiktoken text chunking
│   ├── enrichment.py          # LLM headline + summary per chunk
│   ├── scanner.py             # file listing + change detection
│   ├── sync.py                # incremental sync (new/modified/deleted)
│   ├── scheduler.py           # background scheduled sync
│   ├── retry.py               # exponential backoff decorator
│   ├── folder_registry.py     # SQLite folder tracking
│   └── pipeline.py            # scan → convert → chunk → enrich → embed → store
├── llm/
│   ├── base.py                # LLMProvider protocol
│   ├── factory.py             # create_llm("provider/model") factory
│   ├── ollama_provider.py     # Ollama /api/chat with JSON mode
│   └── anthropic_provider.py  # Anthropic SDK
├── search/
│   ├── pipeline.py            # rewrite → dual search → merge → rerank
│   ├── rewriter.py            # query rewriting with fallback
│   ├── merge.py               # result dedup by source+chunk_index
│   └── reranker.py            # LLM reranking
├── mcp_server/
│   ├── server.py              # FastMCP server (stdio)
│   └── tools.py               # SearchService → SearchPipeline
├── eval/
│   ├── metrics.py             # MRR, nDCG, keyword coverage
│   ├── judge.py               # LLM-as-judge scoring
│   ├── runner.py              # run evaluation
│   ├── compare.py             # multi-config comparison
│   └── test_dataset.jsonl     # 34 curated test questions
├── web/
│   ├── app.py                 # FastAPI server + uvicorn entry
│   ├── routes.py              # REST API endpoints
│   ├── dependencies.py        # shared instances (DI)
│   ├── progress.py            # ingest progress tracking
│   ├── ingest_queue.py        # priority queue with cancel/pause/resume
│   └── static/                # SPA (HTML/CSS/JS)
└── tests/                     # 114 unit + integration tests
```
