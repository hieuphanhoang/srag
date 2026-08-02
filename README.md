# SRAG — Semantic Retrieval-Augmented Generation

[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://python.org)

## Overview

SRAG is a standalone semantic search system: register local folders, it converts/chunks/embeds their documents into its own ChromaDB, and you search across all of them from a web UI, a REST API, or an MCP server (so AI assistants like Claude Code can search your documents directly). It creates its own ChromaDB on first run — no external database required.

## Tech Stack

No agent framework, LLM calls go directly through the official SDKs, behind a small hand-rolled provider abstraction (`llm/base.py` + `llm/factory.py`) that picks Ollama or Anthropic per pipeline stage from a `"provider/model"` string in `config.yaml`.

| Concern | Library |
|---------|---------|
| Web server / API | FastAPI + uvicorn |
| Vector store | ChromaDB (embedded, multi-collection) |
| LLM providers | `anthropic` and `ollama` SDKs directly |
| Document conversion | `markitdown` CLI (PDF/DOCX/PPTX/XLSX) |
| Tokenization/chunking | `tiktoken` |
| AI-assistant integration | `mcp` (Model Context Protocol) SDK |

## Quick Start

### Prerequisites
- **Ollama** installed and running, with an embedding model pulled (`ollama pull qwen3-embedding:8b`)
- Everything else (Python, `uv`) is bootstrapped by the installer below

### Install & Run

```bash
install.bat      # Windows — one-time setup (uv, dependencies, Ollama model, MCP registration)
./install.sh      # macOS/Linux — same, one-time

run.bat           # Windows — start the web UI
./run.sh          # macOS/Linux — start the web UI
```

Open http://localhost:9000 for the dashboard.

## Features

- **Document Ingestion** — PDF, DOCX, PPTX, XLSX, TXT, MD, EPUB, HTML, PNG/JPG (OCR requires the proprietary ABBYY SDK, not bundled)
- **Semantic Search** — multi-collection ChromaDB (one collection per registered folder) with optional LLM query rewrite and reranking
- **MCP Server** — exposes `srag_search`, `srag_ingest_file`, `srag_ingest_folder`, `srag_stats`, `srag_health` for AI assistants
- **Web UI** — Dashboard, Local Folders, Search, Settings
- **Configurable** — full YAML config with `SRAG_*` env var overrides and a runtime `PATCH /api/config`

## Ingest Pipeline

```
register folder → queue → scan → [convert → chunk] per file → (optional enrich) → batch embed → upsert
```

A single background worker processes one folder at a time (FIFO; `prioritize()` can preempt the running job — it pauses and re-queues automatically, safe because re-ingesting is idempotent). For each file:

1. **Convert** — `.txt`/`.md` read directly; PDF/DOCX/PPTX/XLSX go through the `markitdown` **CLI** as a subprocess (deliberately not its Python API, which mis-extracts text — reversed — on some watermarked PDFs); EPUB/HTML use their own converters; image OCR needs the (not bundled) proprietary ABBYY SDK.
2. **Chunk** — token-based splitting (`tiktoken`), 768 tokens with 64 overlap by default.
3. **Enrich** *(optional, off by default)* — a per-chunk LLM call adds headline/summary metadata.
4. **Embed** — batched calls to Ollama's embedding API.
5. **Upsert** — into a per-folder ChromaDB collection (name = deterministic hash of the folder's absolute path), keyed by a deterministic chunk ID (hash of source path + chunk index + content) — re-ingesting an unchanged file **upserts**, never duplicates.

A file that fails to convert, or produces zero chunks, is recorded in the task's error list rather than silently skipped. Same pipeline runs whether triggered from the web UI, an MCP tool, or the CLI (`uv run python -m ingest.pipeline <folder>`). Full walkthrough in [USAGE.md](USAGE.md).

### Comparable Frameworks

This is a hand-rolled version of a well-known pattern, not a novel one — no framework dependency, but the stages above map directly onto:

| SRAG stage | LlamaIndex equivalent | LangChain equivalent |
|---|---|---|
| Convert | `Reader` / `SimpleDirectoryReader` | `DocumentLoader` |
| Chunk | `NodeParser` / `SentenceSplitter` | `TextSplitter` |
| Enrich | `MetadataExtractor` | (no direct equivalent — usually custom) |
| Embed | `embed_model` in `IngestionPipeline` | `Embeddings.embed_documents()` |
| Upsert | `VectorStoreIndex` | `VectorStore.add_documents()` |

[LlamaIndex](https://www.llamaindex.ai/)'s `IngestionPipeline` class is the closest one-to-one match — it's built around exactly this transformation sequence. [Haystack](https://haystack.deepset.ai/) has an equivalent component-pipeline model too. LangGraph is a different category — it's for stateful, branching *agent* orchestration (loops, tool-calling, multi-step decisions), not linear ETL, so it isn't really an alternative to the ingest side specifically. The part of SRAG that *is* agent-facing is the MCP server (external AI assistants call into it) — that's a consumer of the pipeline's output, not built with an agent framework itself.

## Project Structure

```
srag/
├── config.yaml                 # settings
├── install.bat / install.sh    # one-time environment setup
├── run.bat / run.sh             # start the web UI or MCP server
├── core/                        # foundation modules, imported everywhere else
│   ├── config.py                #   settings loader
│   ├── models.py                #   SearchResult, ChunkWithMetadata
│   ├── log.py                   #   logging setup (console + JSON-lines file)
│   └── embedding.py              #   Ollama embedding client
├── store/                       # ChromaDB wrapper (multi-collection)
├── ingest/                      # scan → convert → chunk → embed → store (+ sync, retry)
├── llm/                         # Ollama/Anthropic providers + factory + enrichment
├── search/                      # query rewrite, result merge, reranking
├── mcp_server/                  # MCP tools & stdio server
├── web/                         # FastAPI server, REST API, SPA UI
└── tests/                       # unit + integration tests (272 total)
```

## Configuration

See `config.yaml` for all settings. Key ones:

| Setting | Default | Description |
|---------|---------|-------------|
| `chromadb_path` | `${LOCALAPPDATA}/srag/chromadb` | ChromaDB path (auto-created; resolves correctly on macOS/Linux too) |
| `ollama_url` | `http://localhost:11434` | Ollama server URL |
| `embedding_model` | `qwen3-embedding-8b` | Embedding model name |
| `server.port` | `9000` | Web server port |

For full documentation, see [USAGE.md](USAGE.md).


## Contributing

Pull requests welcome! Please run tests before submitting:

```bash
uv run pytest tests/ -m "not integration"   # unit tests, no external services
uv run pytest tests/ -m integration          # integration tests, needs a running Ollama
```
