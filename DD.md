# Detail Design — SRAG

## DD-01: Config Loading Sequence (`config.py`)

**Focus:** YAML parse → env var expansion → env overrides → required field validation → dataclass construction

### Sequence

```
load_config(path)
  │
  ├─ 1. Read YAML
  │      yaml.safe_load(file) → raw dict (or {} if empty)
  │
  ├─ 2. Expand env vars (recursive)
  │      _expand_recursive(raw)
  │        └─ for every str value: re.sub(r"\$\{(\w+)\}", os.environ[name], value)
  │           raises ValueError if variable is not set
  │
  ├─ 3. Apply env overrides
  │      _apply_env_overrides(raw)
  │        └─ for each SRAG_* env var present in os.environ:
  │             flat target  → raw[key] = value
  │             tuple target → raw.setdefault(section, {})[key] = value
  │
  ├─ 4. Validate required fields
  │      _REQUIRED_FIELDS = ["chromadb_path"]
  │      raises ValueError if field is missing or empty string
  │
  └─ 5. Build dataclass tree
         _build_config(raw)
           └─ extracts each section dict from raw (default {})
              constructs nested dataclasses with explicit field casting:
                int(), _parse_bool(), str()
```

### `_parse_bool(value, default)` logic

| Input type | Rule |
|---|---|
| `bool` | returned as-is |
| `str` | `False` if value.lower().strip() in `{"false","no","0","off"}`, else `True` |
| other non-None | `bool(value)` |
| `None` | returns `default` |

### `_ENV_OVERRIDES` map

| Env var | Target |
|---|---|
| `SRAG_CHROMADB_PATH` | `chromadb_path` (flat) |
| `SRAG_CHROMADB_TENANT` | `chromadb_tenant` (flat) |
| `SRAG_CHROMADB_DATABASE` | `chromadb_database` (flat) |
| `SRAG_OLLAMA_URL` | `ollama_url` (flat) |
| `SRAG_EMBEDDING_MODEL` | `embedding_model` (flat) |
| `SRAG_LOG_LEVEL` | `("logging", "level")` (nested) |
| `SRAG_SERVER_PORT` | `("server", "port")` (nested) |

### Notes

- Env overrides run **after** YAML expansion, so they can overwrite values that were already env-expanded in YAML.
- `chromadb_path` supports `${LOCALAPPDATA}` syntax in `config.yaml`; the expansion in step 2 resolves it before validation.
- All numeric fields are cast with `int()` in `_build_config` — the env override values arrive as strings and are cast at construction time.

## DD-02: ChromaDB Cross-Collection Search Algorithm (`store/chromadb_store.py`)

**Focus:** Per-collection query → merge by distance → global top-K sort

### `search(query_embedding, top_k, collection_names)` sequence

```
search(query_embedding, top_k=10)
  │
  ├─ 1. Resolve collection list
  │      if collection_names is None:
  │        collection_names = [c["name"] for c in list_collections()]
  │
  ├─ 2. Per-collection vector search (parallel loop)
  │      all_results: list[SearchResult] = []
  │      for name in collection_names:
  │        col = get_collection(name)
  │        if col.count() == 0: skip
  │        results = col.query(
  │          query_embeddings=[query_embedding],
  │          n_results=min(top_k, col.count()),
  │          include=["documents", "metadatas", "distances"]
  │        )
  │        for doc, meta, dist in zip(docs, metas, dists):
  │          all_results.append(SearchResult(text, metadata, distance, collection))
  │
  ├─ 3. Global merge & deduplicate
  │      (implicit: all_results grows with each collection's results)
  │
  └─ 4. Global top-K sort by distance (ascending)
         all_results.sort(key=lambda r: r.distance)
         return all_results[:top_k]
```

### Key points

- **Per-collection request limit:** each collection is queried for `min(top_k, col.count())` results. This prevents overfetching but can result in fewer than `top_k` total results across all collections.
- **No dedup:** if a chunk appears in multiple collections, it appears multiple times in results. (By design: different collections are separate datasets.)
- **Distance metric:** collections **must all use the same `hnsw:space`** for the global merge-sort to be valid. All collections are created with `cosine` via `get_collection` (see FD-16 / ISSUE #35). Lower distance = better. Mixing metrics (e.g. `l2` with `cosine`) in one sort systematically mis-ranks results, which is why the space is set explicitly at creation rather than left to ChromaDB's `l2` default.
- **Scope:** srag owns this ChromaDB exclusively, so "all collections" means only srag's own data — no filtering is needed to keep another application's collections out of the results.
- **Error handling:** if a collection query fails (e.g., collection corrupted), log a warning and skip that collection; do not fail the entire search.

### Result structure

```python
SearchResult(
  text: str,
  metadata: dict,  # {source, chunk_index, total_chunks, filename, ...}
  distance: float,
  collection: str  # collection name for provenance
)
```

## DD-03: Embedding Batch Flow (`embedding.py`)

**Focus:** Ollama `/api/embed` call, input batching

### `embed(text)` — single text

```
embed(text: str) → list[float]
  │
  └─ ollama_client.embed(model=self._model, input=text)
       └─ POST /api/embed with single string input
            └─ response["embeddings"][0] → embedding vector (dim=4096 for qwen3-embedding)
```

### `embed_batch(texts)` — multiple texts

```
embed_batch(texts: list[str]) → list[list[float]]
  │
  └─ ollama_client.embed(model=self._model, input=texts)
       └─ POST /api/embed with list[str] input
            └─ response["embeddings"] → list of vectors, one per input text
                 (parallel embedding at the Ollama service level)
```

### Integration with pipeline

In `ingest/pipeline.py` (line 152–163), texts are embedded in batches of 50 to amortize HTTP round-trips:

```python
for batch_start in range(0, len(texts_to_embed), batch_size=50):
  batch_end = min(batch_start + batch_size, len(texts_to_embed))
  embeddings = embed_with_retry(texts_to_embed[batch_start:batch_end])
  upsert_with_retry(collection_name, ids, documents, embeddings, metadatas)
```

### Notes

- **Batching responsibility:** the caller (pipeline, sync, evaluation) decides batch size; `Embedder` simply forwards the list to Ollama.
- **Retry wrapping:** `embed_batch` is wrapped with `@retry(max_retries=3, base_delay=1.0)` at the pipeline level, so transient Ollama failures trigger exponential backoff.
- **Vector dimension:** fixed by the model (4096 for qwen3-embedding); mismatch between index and embedding model will cause upsert failures.

## DD-04: Chunking Algorithm (`ingest/chunker.py`)

**Focus:** Token-based sliding window: start=0, step=max_tokens-overlap, detokenize per window

### `chunk_text(text, max_tokens=512, overlap=50)` sequence

```
chunk_text(text, max_tokens, overlap)
  │
  ├─ 0. Input validation
  │      if not text or not text.strip(): return []
  │
  ├─ 1. Tokenize
  │      tokens = _tokenize(text)  → list of token IDs (tiktoken) or words (fallback)
  │      total = len(tokens)
  │
  ├─ 2. Check single-chunk case
  │      if total <= max_tokens: return [text]  (no chunking needed)
  │
  ├─ 3. Sliding window loop
  │      chunks = []
  │      step = max_tokens - overlap
  │      start = 0
  │      
  │      while start < total:
  │        end = min(start + max_tokens, total)
  │        chunk_tokens = tokens[start:end]
  │        chunk_text = _detokenize(chunk_tokens)
  │        chunks.append(chunk_text)
  │        
  │        if end >= total: break  (last chunk, don't increment)
  │        start += step
  │
  └─ 4. Return chunks
```

### Example walkthrough (max_tokens=512, overlap=50)

- **Total tokens:** 1200, **step:** 462
- **Chunk 1:** tokens[0:512] → detokenize → 512 tokens
- **Chunk 2:** tokens[462:974] → detokenize → 512 tokens (50 overlap with chunk 1)
- **Chunk 3:** tokens[924:1200] → detokenize → 276 tokens (remainder)

### Tokenization fallback

| Condition | Tokenizer | `_tokenize()` | `_detokenize()` |
|---|---|---|---|
| `tiktoken` available | `cl100k_base` encoding | `enc.encode(text)` → list[int] | `enc.decode(tokens)` |
| Not available | Whitespace split | `text.split()` → list[str] | `" ".join(tokens)` |

### Notes

- **Overlap semantics:** tokens from the end of chunk N appear at the start of chunk N+1, providing context continuity.
- **Window movement:** `start` increments by `max_tokens - overlap`, which shifts the window forward by `step` tokens each iteration.
- **Last chunk:** once `end >= total`, the loop breaks to avoid an empty iteration.
- **No minimum chunk size:** a final chunk can be arbitrarily small (even 1 token) if it's the remainder.

## DD-05: Collection Naming Convention (`ingest/folder_registry.py`)

**Focus:** SHA-256(normcase(abspath(folder_path))) — bare digest, no prefix

### `collection_name_for_path(folder_path)` algorithm

```
collection_name_for_path(folder_path: str) → str
  │
  ├─ 1. Normalize path to absolute canonical form
  │      normalized = os.path.normcase(os.path.abspath(folder_path))
  │      └─ normcase: lowercase on Windows, unchanged on Unix
  │      └─ abspath: resolve relative paths, . and .. segments
  │
  └─ 2. Compute SHA-256 hash and return it
         return hashlib.sha256(normalized.encode()).hexdigest()
         └─ deterministic, 64-char hex string, e.g. "d63084309f5ad229..."
```

### Rationale

- **No prefix:** srag owns its ChromaDB outright (its own path + default tenant/database), so the collection namespace is shared with nothing and needs no disambiguating prefix. Every collection in the database belongs to srag.
- **Path normalization:** ensures the same folder always produces the same collection name, regardless of how it's specified (relative, absolute, symlinks, case variants on Windows).
- **SHA-256 hash:** stable, human-opaque name that avoids collisions; matches the digest already used by `generate_chunk_id` (FD-10). A hash is used (vs. folder name) to handle long paths and special characters gracefully. The 64-char digest is well within ChromaDB's collection-name limits (verified on chromadb 1.5.9).

### Example

```python
# On Windows:
collection_name_for_path("C:\\Users\\Alice\\My Documents")
  → SHA256("c:\\users\\alice\\my documents")
  → "f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6..."  (64 hex chars)

# Same folder, different notation:
collection_name_for_path("C:/Users/Alice/My Documents")
  → same digest (normcase handles case & slash)
```

### Storage

When a folder is registered (FD-09 `register()`), the collection name is computed and stored in the `folders` table:

```sql
INSERT INTO folders (path, collection_name, created_at)
  VALUES ('C:\\Users\\Alice\\My Documents', 'f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6…', now)  -- bare 64-char SHA-256, no prefix
```

Future ingest/sync operations retrieve the collection name from the registry rather than recomputing it, ensuring consistency even if the path representation changes (relative → absolute, etc.).

## DD-06: Folder Registry SQLite Schema (`ingest/folder_registry.py`)

**Focus:** 3 tables (folders, files, file_errors), migrations, UPSERT conflict resolution

### Table: `folders`

```sql
CREATE TABLE IF NOT EXISTS folders (
    path TEXT PRIMARY KEY,
    collection_name TEXT UNIQUE NOT NULL,
    doc_count INTEGER DEFAULT 0,
    chunk_count INTEGER DEFAULT 0,
    last_indexed TEXT,
    status TEXT DEFAULT 'registered',
    created_at TEXT NOT NULL,
    excluded INTEGER DEFAULT 0,
    single INTEGER DEFAULT 0
);
```

| Column | Type | Purpose | Notes |
|---|---|---|---|
| `path` | TEXT PRIMARY KEY | Folder absolute path | Unique per folder; ingest/web routes resolve all paths to absolute form before querying |
| `collection_name` | TEXT UNIQUE | ChromaDB collection name | Computed via `collection_name_for_path()` at registration time; UNIQUE to prevent two folders mapping to the same collection |
| `doc_count` | INTEGER | Number of documents (files) successfully ingested | Updated by `update_status()` after ingest completes |
| `chunk_count` | INTEGER | Total chunks across all docs | Includes chunks from enriched texts |
| `last_indexed` | TEXT | ISO 8601 timestamp | NULL until first ingest; used to report "never indexed" |
| `status` | TEXT | One of `{registered, indexing, ready, missing}` | Set to `registered` on creation; `ready` after successful ingest; `missing` if folder path doesn't exist on disk |
| `created_at` | TEXT | ISO 8601 timestamp of registration | Set at registration time, immutable |
| `excluded` | INTEGER (bool) | 1 if folder is excluded from sync, 0 otherwise | Toggled by `toggle_excluded()` to suspend auto-sync without unregistering |
| `single` | INTEGER (bool) | 1 if folder is in "single mode" (only one folder can be 1 at a time) | Toggled by `toggle_single()` to restrict search to one collection |

### Table: `files`

```sql
CREATE TABLE IF NOT EXISTS files (
    path TEXT PRIMARY KEY,
    folder_path TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    modified_at TEXT NOT NULL,
    chunk_count INTEGER DEFAULT 0,
    last_indexed TEXT
);
```

| Column | Type | Purpose | Notes |
|---|---|---|---|
| `path` | TEXT PRIMARY KEY | File absolute path | Unique identifier for a file |
| `folder_path` | TEXT NOT NULL | Parent folder's absolute path | Foreign key reference (no explicit FK constraint) to `folders.path`; used for cascading deletes and change detection |
| `content_hash` | TEXT | MD5 hash of file contents | Computed via `file_content_hash()` after each ingest; used to detect actual content changes vs. mtime-only updates |
| `modified_at` | TEXT | ISO 8601 mtime from the file's last `stat()` call | Compared with current file mtime (epsilon 0.5s) to detect potential changes; if different, content_hash is checked to confirm |
| `chunk_count` | INTEGER | Number of chunks produced from this file after chunking | Used to estimate chunks removed when file is deleted during sync |
| `last_indexed` | TEXT | ISO 8601 timestamp when this file was last ingested | For future feature: prioritize re-ingest of old files |

### Table: `file_errors`

```sql
CREATE TABLE IF NOT EXISTS file_errors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT NOT NULL,
    folder_path TEXT NOT NULL,
    error_message TEXT NOT NULL,
    timestamp TEXT NOT NULL
);
```

| Column | Type | Purpose | Notes |
|---|---|---|---|
| `id` | INTEGER PRIMARY KEY | Auto-increment error ID | Unused by code; present for audit trail |
| `file_path` | TEXT | Path of the file that failed to process | Not a FK; the file may have been deleted |
| `folder_path` | TEXT | Parent folder | Allows filtering errors by folder |
| `error_message` | TEXT | Exception message from the ingest/sync pipeline | E.g., `"UnicodeDecodeError: 'utf-8' codec..."` |
| `timestamp` | TEXT | ISO 8601 when the error was logged | For sorting and aging out old errors |

### Migrations

```python
_MIGRATIONS = [
    "ALTER TABLE folders ADD COLUMN excluded INTEGER DEFAULT 0",
    "ALTER TABLE folders ADD COLUMN single INTEGER DEFAULT 0",
]
```

Executed in `__init__()` with `try/except sqlite3.OperationalError` to skip if the column already exists. This pattern is used instead of a versioning scheme because:
- New deployments create tables from scratch via `_SCHEMA`, so migrations are not run.
- Existing databases run migrations at connection time; idempotent `ALTER TABLE` statements safely retry.

### UPSERT Pattern (Conflict Resolution)

**File upsert** (FD-09 `upsert_file()`):

```sql
INSERT INTO files (path, folder_path, content_hash, modified_at, chunk_count, last_indexed)
  VALUES (?, ?, ?, ?, ?, ?)
ON CONFLICT(path) DO UPDATE SET
  content_hash=excluded.content_hash,
  modified_at=excluded.modified_at,
  chunk_count=excluded.chunk_count,
  last_indexed=excluded.last_indexed
```

- **Semantics:** if the file was already tracked, update its hash, mtime, and chunk count; if new, insert.
- **Note:** `folder_path` is NOT updated on conflict. This ensures a file cannot be moved between folders without explicit deletion+reinsertion.

**Folder insert** (FD-09 `register()`):

```sql
INSERT INTO folders (path, collection_name, created_at) VALUES (?, ?, ?)
```

- No `ON CONFLICT` clause. If a folder is already registered, `sqlite3.IntegrityError` is caught and re-raised as `ValueError("Folder already registered")`.

### Notes

- **No explicit foreign keys:** SQLite FK enforcement is not enabled in this codebase. Cascades are handled in Python (e.g., `unregister()` deletes from all three tables).
- **String IDs, no surrogate keys:** the tables use natural keys (path strings, not integer IDs), which is intentional for auditability (no need to cross-reference IDs).
- **Timestamp format:** all timestamps are ISO 8601 strings (via `datetime.now(timezone.utc).isoformat()`), not Unix timestamps. This is human-readable and timezone-safe.

## DD-07: Change Detection Algorithm (`ingest/change_detector.py`)

**Focus:** Scan disk → compare with tracked files → mtime epsilon (0.5s) → content hash fallback → ChangeSet

### `detect_changes(folder_path)` sequence

```
detect_changes(folder_path: str) → ChangeSet
  │
  ├─ 1. Scan disk for current files
  │      current_files = _scan_folder(folder_path)
  │        └─ walk folder recursively
  │        └─ collect file paths and their disk stat (mtime, size)
  │        └─ skip hidden files and common exclude patterns
  │
  ├─ 2. Retrieve tracked files from registry
  │      tracked_files = registry.list_files(folder_path)
  │        └─ query `files` table WHERE folder_path = ?
  │        └─ build dict: {path → (content_hash, modified_at, chunk_count)}
  │
  ├─ 3. Classify files
  │      │
  │      ├─ A. Deleted: in tracked_files but not in current_files
  │      │      for each: ChangeSet.add_deleted(path, chunk_count)
  │      │
  │      ├─ B. New: in current_files but not in tracked_files
  │      │      for each: ChangeSet.add_new(path)
  │      │
  │      └─ C. Modified or unchanged: in both
  │             for each path in both:
  │               disk_mtime = current_files[path].mtime
  │               tracked_mtime = tracked_files[path].modified_at
  │               
  │               if |disk_mtime - tracked_mtime| <= epsilon (0.5s):
  │                 → assume unchanged, skip
  │               else:
  │                 → mtime changed, check content hash
  │                   disk_hash = file_content_hash(path)
  │                   tracked_hash = tracked_files[path].content_hash
  │                   
  │                   if disk_hash == tracked_hash:
  │                     → content unchanged (mtime touched but content same)
  │                     → skip
  │                   else:
  │                     → content modified
  │                     → ChangeSet.add_modified(path)
  │
  └─ 4. Return ChangeSet
         ChangeSet(new=[], modified=[], deleted=[...])
```

### ChangeSet structure

```python
@dataclass
class ChangeSet:
    new: list[str]      # new file paths
    modified: list[str] # modified file paths
    deleted: list[DeletedFile]  # deleted files with metadata
    
@dataclass
class DeletedFile:
    path: str
    chunk_count: int    # used to estimate chunks removed from index
    folder_path: str
```

### Key decision points

| Scenario | Decision | Rationale |
|---|---|---|
| File mtime changed, content same | Skip (unchanged) | Touch/metadata operations shouldn't trigger re-ingest |
| File mtime unchanged | Skip (unchanged) | Assume no concurrent external modifications during scan |
| File mtime changed, content changed | Mark modified | Clear evidence of edits; must re-ingest |
| Disk file newer than tracked (by >0.5s) | Check content hash | Detect actual changes while tolerating minor clock skew |

### Mtime epsilon rationale

- **0.5 seconds:** accommodates file system timestamp resolution (NTFS/ext4 granularity ~1ms, but distributed systems and network shares may have clock skew). A 0.5s delta is conservative enough to avoid false positives while still catching real edits.
- **Fallback to content hash:** if mtime changes but content is identical, the hash comparison proves no actual change occurred.

### Error handling

- **File deleted between scan and hash check:** `file_content_hash()` will raise `FileNotFoundError`. Catch and treat as deleted (remove from ChangeSet.modified, add to ChangeSet.deleted).
- **Permission denied:** `file_content_hash()` will raise `PermissionError`. Log warning and skip (treat as unchanged); do not block the entire scan.
- **Corrupted mtime (future/far past):** No validation; assume system clock is correct. If skew is extreme, content hash fallback catches it.

### Integration with sync pipeline (FD-10)

After `detect_changes()` returns:
1. Delete chunks for files in `ChangeSet.deleted` (query ChromaDB by chunk metadata).
2. Re-ingest files in `ChangeSet.new` and `ChangeSet.modified` (same flow: convert → chunk → embed → upsert).
3. Update registry: mark deleted files as absent, upsert new/modified file tracking.

## DD-08: Ingest Pipeline Sequence (`ingest/pipeline.py`)

**Focus:** scan → per-file: convert → chunk → [enrich] → embed (batched) → upsert, with cancel checkpoints

### `ingest(folder_path, enrich=False, cancel_flag=None)` sequence

```
ingest(folder_path, enrich=False, cancel_flag=None) → IngestResult
  │
  ├─ 0. Setup
  │      folder = registry.get_folder(folder_path)  # raises ValueError if not registered
  │      collection = store.get_collection(folder.collection_name)
  │      registry.update_status(folder_path, 'indexing')
  │
  ├─ 1. Scan folder for files
  │      file_paths = _scan_folder(folder_path)
  │        └─ recursive walk, same as change_detector
  │        └─ skip hidden, exclude patterns
  │
  ├─ 2. For each file: convert & chunk
  │      chunks_by_file = {}  # {file_path → [Chunk]}
  │      
  │      for file_path in file_paths:
  │        checkpoint: if cancel_flag is set, raise IngestCancelled
  │        
  │        try:
  │          text = converter.convert(file_path)
  │          chunks = chunker.chunk_text(text)
  │          chunks_by_file[file_path] = chunks
  │        except Exception as e:
  │          registry.log_file_error(file_path, folder_path, str(e))
  │          continue  # skip this file, process next
  │
  ├─ 3. Optional enrichment (if enrich=True)
  │      texts_to_enrich = []
  │      chunk_to_text_idx = {}  # map back to original chunk
  │      
  │      for file_path, chunks in chunks_by_file.items():
  │        for idx, chunk in enumerate(chunks):
  │          texts_to_enrich.append(chunk.text)
  │          chunk_to_text_idx[len(texts_to_enrich)-1] = (file_path, idx)
  │      
  │      checkpoint: if cancel_flag is set, raise IngestCancelled
  │      
  │      enriched_texts = enricher.enrich_batch(
  │        texts_to_enrich,
  │        cancel_flag=cancel_flag
  │      )
  │        └─ see DD-10, DD-11
  │      
  │      # Merge enriched metadata back into chunks
  │      for idx, enriched in enumerate(enriched_texts):
  │        if enriched is None: continue  # enrichment failed
  │        file_path, chunk_idx = chunk_to_text_idx[idx]
  │        chunks_by_file[file_path][chunk_idx].metadata['enriched'] = enriched
  │
  ├─ 4. Batch embed (50 chunks/batch)
  │      all_chunks = []
  │      for file_path, chunks in chunks_by_file.items():
  │        all_chunks.extend([(file_path, c) for c in chunks])
  │      
  │      embeddings_result = {}  # {chunk_id → embedding}
  │      
  │      for batch_start in range(0, len(all_chunks), 50):
  │        checkpoint: if cancel_flag is set, raise IngestCancelled
  │        
  │        batch_end = min(batch_start + 50, len(all_chunks))
  │        batch = all_chunks[batch_start:batch_end]
  │        
  │        texts = [chunk.text for _, chunk in batch]
  │        embeddings = embedding.embed_batch_with_retry(texts)
  │        
  │        for (file_path, chunk), emb in zip(batch, embeddings):
  │          chunk_id = _make_chunk_id(file_path, chunk.index)
  │          embeddings_result[chunk_id] = emb
  │
  ├─ 5. Upsert to ChromaDB
  │      ids = []
  │      documents = []
  │      embeddings = []
  │      metadatas = []
  │      
  │      for file_path, chunks in chunks_by_file.items():
  │        for chunk in chunks:
  │          chunk_id = _make_chunk_id(file_path, chunk.index)
  │          ids.append(chunk_id)
  │          documents.append(chunk.text)
  │          embeddings.append(embeddings_result[chunk_id])
  │          metadatas.append({
  │            'filename': os.path.basename(file_path),
  │            'source': file_path,
  │            'chunk_index': chunk.index,
  │            'total_chunks': len(chunks),
  │            'parent_document_id': _make_doc_id(file_path),
  │            'enriched': chunk.metadata.get('enriched', None)
  │          })
  │      
  │      checkpoint: if cancel_flag is set, raise IngestCancelled
  │      
  │      collection.upsert(
  │        ids=ids,
  │        documents=documents,
  │        embeddings=embeddings,
  │        metadatas=metadatas
  │      )
  │
  ├─ 6. Update registry
  │      for file_path in chunks_by_file.keys():
  │        content_hash = file_content_hash(file_path)
  │        chunk_count = len(chunks_by_file[file_path])
  │        registry.upsert_file(
  │          path=file_path,
  │          folder_path=folder_path,
  │          content_hash=content_hash,
  │          modified_at=file_stat(file_path).mtime,
  │          chunk_count=chunk_count,
  │          last_indexed=now()
  │        )
  │      
  │      doc_count = len(chunks_by_file)
  │      chunk_count = sum(len(c) for c in chunks_by_file.values())
  │      registry.update_status(
  │        folder_path,
  │        status='ready',
  │        doc_count=doc_count,
  │        chunk_count=chunk_count,
  │        last_indexed=now()
  │      )
  │
  └─ 7. Return result
         IngestResult(
           folder_path=folder_path,
           files_indexed=len(chunks_by_file),
           chunks_indexed=chunk_count,
           errors=registry.get_errors(folder_path),
           elapsed_seconds=...
         )
```

### Cancel checkpoints

- **Purpose:** allow cancellation during long-running operations without leaving the database in an inconsistent state.
- **Placement:** before expensive operations (file I/O, conversion, embedding, upsert) and after batches complete.
- **Semantics:** if `cancel_flag` is set (a `threading.Event` or similar), raise `IngestCancelled` exception. The caller catches it and may retry or report the cancellation to the UI.
- **Batch-level atomicity:** embedding and upsert are batched; once a batch starts, it runs to completion. Cancelling mid-batch will cause that batch to fail (e.g., partial embeddings). The next retry starts from the beginning of the pipeline.

### Chunk ID scheme

```python
def _make_chunk_id(file_path: str, chunk_index: int) -> str:
    return f"{_make_doc_id(file_path)}#{chunk_index}"
    
def _make_doc_id(file_path: str) -> str:
    # Deterministic stable ID based on file path + content hash
    return hashlib.md5(f"{file_path}".encode()).hexdigest()
```

- **Rationale:** stable IDs ensure re-ingesting the same file produces the same chunk IDs, enabling deduplication and updates in ChromaDB.

### Error handling & recovery

| Error | Handling | Rationale |
|---|---|---|
| File conversion **raises** | Log in `file_errors` table, record in `result.errors`, skip file, continue | Allow partial ingests; one bad file must not abort the folder |
| File conversion **returns `None`** (unsupported / empty / missing markitdown extra) | Append `"<path>: conversion failed or unsupported format"` to `result.errors` **and** `progress.errors`, then continue | ISSUE #38 — this path used to `continue` silently, so a folder whose every file failed reported `status=ready, 0 chunks, 0 errors`. Failures must reach the registry, `GET /api/folders/{path}/errors`, and the UI. |
| Conversion succeeds but yields no chunks | Append `"<path>: converted but produced no text chunks"` to `result.errors` / `progress.errors`, continue | Same rationale — an empty extraction is a silent data-loss signal, not a success |
| Enrichment fails for a chunk | Return `None` for that chunk's enrichment, skip merge | Enrichment is optional; don't block on failures |
| Embedding batch fails | Retry with exponential backoff (DD-12); if max retries exceeded, raise | Transient Ollama errors should recover; permanent failures escalate |
| Upsert fails | Log error, raise to caller | ChromaDB connection loss is unrecoverable at this level |
| Cancel flag set | Raise `IngestCancelled` immediately | User cancellation takes priority; registry status remains 'indexing' until explicit retry |

### Notes

- **File order:** processed in arbitrary directory-walk order; no sorting.
- **Enrichment scope:** only runs if `enrich=True`. Allows fast re-ingest without enrichment overhead.
- **Batch size:** 50 chunks for embedding to balance round-trip latency vs. Ollama memory use.
- **No transaction:** registry updates are per-file, not atomic with ChromaDB upsert. If a crash occurs between upsert and registry update, the next ingest will detect the drift and re-process (idempotent by chunk ID).

## DD-09: Sync Pipeline Sequence (`ingest/sync.py`)

**Focus:** scan_changes → delete removed chunks → process new/modified → update file tracking → update folder status

### `sync(folder_path, enrich=False, cancel_flag=None)` sequence

```
sync(folder_path, enrich=False, cancel_flag=None) → SyncResult
  │
  ├─ 0. Setup
  │      folder = registry.get_folder(folder_path)
  │      if folder.status == 'missing':
  │        → raise ValueError("Folder does not exist")
  │      
  │      if not folder.excluded:
  │        registry.update_status(folder_path, 'indexing')
  │      collection = store.get_collection(folder.collection_name)
  │
  ├─ 1. Detect changes
  │      changeset = change_detector.detect_changes(folder_path)
  │        → ChangeSet(new=[...], modified=[...], deleted=[DeletedFile(...)])
  │
  ├─ 2. Delete removed chunks
  │      for deleted_file in changeset.deleted:
  │        doc_id = _make_doc_id(deleted_file.path)
  │        # Query ChromaDB for all chunks with this parent_document_id
  │        results = collection.query(
  │          where={'parent_document_id': doc_id},
  │          include=['documents']
  │        )
  │        chunk_ids = [rid for rid in results.ids[0]]
  │        
  │        if chunk_ids:
  │          collection.delete(ids=chunk_ids)
  │      
  │      # Update registry: mark file as deleted (remove from files table)
  │      for deleted_file in changeset.deleted:
  │        registry.delete_file(deleted_file.path)
  │
  ├─ 3. Process new & modified files
  │      files_to_process = changeset.new + changeset.modified
  │      
  │      for file_path in files_to_process:
  │        checkpoint: if cancel_flag is set, raise SyncCancelled
  │        
  │        try:
  │          text = converter.convert(file_path)
  │          chunks = chunker.chunk_text(text)
  │          
  │          # Enrich if enabled
  │          if enrich:
  │            texts = [c.text for c in chunks]
  │            enriched = enricher.enrich_batch(texts, cancel_flag=cancel_flag)
  │            for idx, c in enumerate(chunks):
  │              if enriched[idx]:
  │                c.metadata['enriched'] = enriched[idx]
  │          
  │          # Embed batch (50 chunks)
  │          embeddings = []
  │          for batch_start in range(0, len(chunks), 50):
  │            batch_end = min(batch_start + 50, len(chunks))
  │            texts = [chunks[i].text for i in range(batch_start, batch_end)]
  │            batch_emb = embedding.embed_batch_with_retry(texts)
  │            embeddings.extend(batch_emb)
  │          
  │          # Upsert (replaces old chunks via same chunk IDs)
  │          ids = [_make_chunk_id(file_path, i) for i in range(len(chunks))]
  │          docs = [c.text for c in chunks]
  │          metas = [{
  │            'filename': os.path.basename(file_path),
  │            'source': file_path,
  │            'chunk_index': i,
  │            'total_chunks': len(chunks),
  │            'parent_document_id': _make_doc_id(file_path),
  │            'enriched': chunks[i].metadata.get('enriched')
  │          } for i in range(len(chunks))]
  │          
  │          collection.upsert(ids=ids, documents=docs, embeddings=embeddings, metadatas=metas)
  │          
  │          # Update file tracking
  │          registry.upsert_file(
  │            path=file_path,
  │            folder_path=folder_path,
  │            content_hash=file_content_hash(file_path),
  │            modified_at=file_stat(file_path).mtime,
  │            chunk_count=len(chunks),
  │            last_indexed=now()
  │          )
  │        
  │        except Exception as e:
  │          registry.log_file_error(file_path, folder_path, str(e))
  │          continue
  │
  ├─ 4. Update folder status
  │      # Recount docs & chunks from registry
  │      files = registry.list_files(folder_path)
  │      doc_count = len(files)
  │      chunk_count = sum(f.chunk_count for f in files)
  │      
  │      registry.update_status(
  │        folder_path,
  │        status='ready',
  │        doc_count=doc_count,
  │        chunk_count=chunk_count,
  │        last_indexed=now()
  │      )
  │
  └─ 5. Return result
         SyncResult(
           folder_path=folder_path,
           new_files=len(changeset.new),
           modified_files=len(changeset.modified),
           deleted_files=len(changeset.deleted),
           chunks_indexed=chunk_count,
           errors=registry.get_errors(folder_path),
           elapsed_seconds=...
         )
```

### Differences from ingest (DD-08)

| Aspect | Ingest | Sync |
|---|---|---|
| Pre-condition | Folder must be registered | Folder must exist on disk |
| File discovery | Scan all files | Scan all files, then diff against tracked files |
| Processing | All files in folder | Only new/modified files (skip unchanged) |
| Deletion | N/A (ingest doesn't delete) | Delete chunks for removed files |
| Error recovery | Skip file, continue | Skip file, continue (same) |

### Key optimizations

- **Skips unchanged files:** change detection means we avoid re-converting/chunking/embedding files that haven't changed.
- **Content hash fallback:** mtime epsilon (0.5s) and content hash ensure we don't re-process files that were touched but not edited.
- **Chunk ID stability:** re-upserting with the same chunk ID replaces old embeddings automatically (ChromaDB upsert semantics).

### Registry consistency

After sync completes, the registry reflects the current disk state:
- `files` table contains only files that exist on disk and were successfully processed.
- `file_errors` table accumulates errors from failed conversions.
- `folders` table has updated `doc_count`, `chunk_count`, `last_indexed`, and status.

### Cancel semantics

- **Cancellation during change detection:** scan completes, then raises `SyncCancelled`. No changes are made.
- **Cancellation during file processing:** stops at the next checkpoint. Already-processed files' embeddings are in ChromaDB; already-updated registry entries are committed. Re-running sync resumes from the beginning (safe because chunk IDs are stable).
- **Cancellation during enrichment:** handled by enricher (DD-11). Current batch stops; sync continues to embedding/upsert.

### Notes

- **Folder missing detection:** if folder path doesn't exist on disk at the start, sync raises an error (status remains 'missing'). Callers must re-register the folder or restore the path.
- **No full ingest:** sync is incremental; to re-ingest all files without change detection, call `ingest()` instead (which discards old registrations and re-scans the full folder).
- **Conflict-free upsert:** because chunk IDs are deterministic, re-running sync on the same changeset produces identical results (idempotent).

## DD-10: Enrichment Retry with Backoff (`llm/enricher.py`)

**Focus:** Per-chunk: prompt LLM → parse JSON → on failure: sleep(2^attempt) → max 3 retries

### `enrich(text, cancel_flag=None)` sequence

```
enrich(text: str, cancel_flag=None) → EnrichedMetadata | None
  │
  ├─ 0. Check cancellation
  │      if cancel_flag and cancel_flag.is_set():
  │        raise EnrichmentCancelled
  │
  ├─ 1. Prompt LLM with retry loop
  │      for attempt in range(3):  # 0, 1, 2 (max 3 retries)
  │        try:
  │          checkpoint: if cancel_flag, check again
  │          
  │          response = llm_client.call(
  │            system=ENRICHMENT_SYSTEM_PROMPT,
  │            user_message=f"Enrich this text:\n\n{text}",
  │            json_mode=True,
  │            timeout=30s
  │          )
  │          
  │          # Parse JSON response
  │          enriched_data = json.loads(response.text)
  │          
  │          # Validate schema
  │          if not _is_valid_enrichment(enriched_data):
  │            raise ValueError("Invalid enrichment schema")
  │          
  │          return EnrichedMetadata(**enriched_data)
  │        
  │        except (json.JSONDecodeError, ValueError) as e:
  │          # Validation errors, not transient
  │          logger.warning(f"Enrichment validation failed: {e}")
  │          return None  # give up, don't retry
  │        
  │        except (TimeoutError, ConnectionError, APIError) as e:
  │          # Transient errors, retry with backoff
  │          if attempt < 2:  # not the last attempt
  │            delay = 2 ** attempt  # 1s, 2s, 4s
  │            logger.info(f"Enrichment attempt {attempt+1} failed, retrying in {delay}s: {e}")
  │            
  │            checkpoint: if cancel_flag, check before sleep
  │            sleep(delay)
  │          else:
  │            # Last attempt failed
  │            logger.error(f"Enrichment failed after 3 attempts: {e}")
  │            return None
  │
  └─ 2. Return enriched metadata or None
```

### Exponential backoff schedule

| Attempt | Delay | Cumulative |
|---|---|---|
| 0 (first try) | — | — |
| 1 (1st retry) | 2^0 = 1s | 1s |
| 2 (2nd retry) | 2^1 = 2s | 3s |
| Max total | — | ~3s |

### Error classification

| Error | Type | Action |
|---|---|---|
| JSON decode error | Validation | Return `None` immediately (LLM output malformed) |
| Invalid schema | Validation | Return `None` immediately (LLM response doesn't match spec) |
| Timeout (>30s) | Transient | Retry with backoff |
| Connection error | Transient | Retry with backoff |
| LLM API error (5xx) | Transient | Retry with backoff |
| LLM rate limit (429) | Transient | Retry with backoff |
| Invalid API key | Permanent | Return `None` after 1st attempt (assume key is wrong) |

### EnrichedChunk schema (matches FD-55)

```python
@dataclass
class EnrichedChunk:
    headline: str             # Short descriptive title (max 100 chars)
    summary: str              # 1–2 sentence summary
    original_text: str        # Original chunk text, preserved verbatim
    
    def as_text(self) -> str:
        """Returns headline + summary + original_text for embedding."""
        return f"{self.headline}\n\n{self.summary}\n\n{self.original_text}"
```

EnrichedChunk instances stored in chunk `metadata['enriched']` (or `None` if enrichment failed).

### Cancellation handling

- **Checkpoint before LLM call:** checks `cancel_flag` before making potentially expensive API call.
- **Checkpoint before sleep:** checks `cancel_flag` before entering backoff delay, allowing immediate cancellation without waiting.
- **Semantics:** if cancelled during enrichment, raise `EnrichmentCancelled` and let the caller (DD-08 or DD-09) decide whether to retry or abort.

### Timeout rationale

- **30 seconds per call:** typical LLM response time is <5s; 30s allows for queueing delays and slow network.
- **No per-attempt timeout:** backoff delays are independent of the call timeout. An attempt that takes 30s is counted as 1 attempt, then the backoff timer starts fresh.

### Notes

- **Single-text enrichment:** `enrich()` handles one chunk at a time. Batching is managed by the caller (DD-11).
- **No fallback LLM:** if enrichment fails, metadata is simply `None`; no degraded alternative is attempted.
- **Idempotent enrichment:** the same text always produces the same enrichment (assuming LLM is deterministic), so re-enriching is safe and produces correct results.

## DD-11: Enrichment Batch Concurrency (`llm/enricher.py`)

**Focus:** ThreadPoolExecutor(workers=3), cancel_flag checked between futures

### `enrich_batch(texts, cancel_flag=None)` sequence

```
enrich_batch(texts: list[str], cancel_flag=None) → list[EnrichedChunk | None]
  │
  ├─ 0. Setup thread pool
  │      executor = ThreadPoolExecutor(max_workers=3)
  │      futures = []
  │
  ├─ 1. Submit all tasks
  │      for idx, text in enumerate(texts):
  │        future = executor.submit(
  │          self.enrich,
  │          text,
  │          cancel_flag=cancel_flag
  │        )
  │        futures.append(future)
  │
  ├─ 2. Collect results with cancellation checks
  │      results = []
  │      
  │      for idx, future in enumerate(futures):
  │        checkpoint: if cancel_flag is set, raise EnrichmentCancelled
  │        
  │        try:
  │          result = future.result(timeout=35s)
  │          results.append(result)
  │        except TimeoutError:
  │          logger.error(f"Enrichment for text {idx} timed out")
  │          results.append(None)
  │        except EnrichmentCancelled:
  │          # Propagate cancellation upward
  │          executor.shutdown(wait=False)
  │          raise
  │        except Exception as e:
  │          logger.error(f"Enrichment for text {idx} failed: {e}")
  │          results.append(None)
  │
  └─ 3. Shutdown executor and return
         executor.shutdown(wait=True)
         return results
```

### Concurrency model

- **Workers:** 3 concurrent threads (hardcoded). Rationale:
  - Avoids overwhelming the LLM service with too many simultaneous requests.
  - Provides parallelism within a single folder's ingest (3-5 concurrent API calls).
  - Respects rate limits of typical LLM endpoints (limit ~10 req/s per account).

- **Backpressure:** `ThreadPoolExecutor` internally queues tasks; if all 3 workers are busy, new tasks wait in the queue (no explicit backpressure logic needed).

- **Task submission:** all tasks are submitted upfront (eager), not lazily. This allows the executor to manage queueing internally.

### Cancellation semantics

- **Checkpoint before collecting results:** if `cancel_flag` is set, stop waiting for futures and raise `EnrichmentCancelled`.
- **Graceful shutdown on cancel:** `executor.shutdown(wait=False)` stops accepting new tasks but allows running threads to finish their current `enrich()` call (which checks `cancel_flag` internally via DD-10).
- **Partial results:** if cancellation occurs mid-batch, already-completed enrichments are lost. The caller (DD-08 or DD-09) retries the entire batch on next ingest/sync.

### Timeout handling

- **35 seconds per future:** slightly longer than the 30s LLM timeout in DD-10, to account for internal queuing and thread scheduling delays.
- **Timeout → None:** if a future times out, it's treated as a failed enrichment (logged, return `None`). The chunk is still ingested without enrichment metadata.

### Integration with ingest/sync

Callers (DD-08, DD-09) invoke `enrich_batch()` with a list of texts:

```python
texts = [chunk.text for chunk in chunks]
enriched_list = enricher.enrich_batch(texts, cancel_flag=cancel_flag)

for chunk, enriched in zip(chunks, enriched_list):
  if enriched:
    chunk.metadata['enriched'] = enriched
```

### Notes

- **Thread-safe:** `enrich()` is stateless and thread-safe (no shared state between threads).
- **Graceful degradation:** individual enrichment failures don't block the batch; they simply return `None`, and the chunk is ingested without metadata.
- **No retries at batch level:** retries happen inside `enrich()` (DD-10). If a chunk's enrichment ultimately fails, the batch continues without re-attempting.
- **Executor scope:** a new executor is created per `enrich_batch()` call (not reused). This avoids complexity around shutdown and allows each batch to be independent.

## DD-12: Retry Decorator Implementation (`core/retry.py`)

**Focus:** Decorator factory, exponential delay (base * 2^attempt), configurable exception filter

### `@retry(max_retries=3, base_delay=1.0, exceptions=(Exception,))` decorator

```python
def retry(max_retries=3, base_delay=1.0, exceptions=(Exception,)):
    """
    Retry decorator factory.
    
    Args:
        max_retries: maximum number of retries after initial failure (total attempts = max_retries + 1)
        base_delay: initial backoff delay in seconds (grows exponentially)
        exceptions: tuple of exception types to catch and retry on
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(
                            f"{func.__name__} attempt {attempt+1}/{max_retries+1} failed: {e}. "
                            f"Retrying in {delay}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"{func.__name__} failed after {max_retries+1} attempts: {e}"
                        )
            
            raise last_exception
        
        return wrapper
    return decorator
```

### Exponential backoff formula

```
delay = base_delay * (2 ^ attempt)

Examples with base_delay=1.0:
  attempt 0 (1st try):      — (no delay before first attempt)
  attempt 1 (1st retry):    1.0 * 2^0 = 1.0s
  attempt 2 (2nd retry):    1.0 * 2^1 = 2.0s
  attempt 3 (3rd retry):    1.0 * 2^2 = 4.0s
  total max wait:           1 + 2 + 4 = 7s

Examples with base_delay=0.5:
  attempt 1:                0.5s
  attempt 2:                1.0s
  attempt 3:                2.0s
  total max wait:           3.5s
```

### Usage examples

```python
# Embed batch with transient error retry
@retry(max_retries=3, base_delay=1.0, exceptions=(TimeoutError, ConnectionError))
def embed_batch_with_retry(texts: list[str]) -> list[list[float]]:
    return ollama_client.embed(model=self._model, input=texts)

# Upsert with transient error retry
@retry(max_retries=3, base_delay=1.0, exceptions=(Exception,))
def upsert_with_retry(collection, ids, documents, embeddings, metadatas):
    return collection.upsert(ids=ids, documents=documents, ...)

# Custom exception filter
@retry(max_retries=2, base_delay=0.5, exceptions=(APIError, TimeoutError))
def enrich_text(text: str) -> EnrichedMetadata:
    # Only retries on APIError or TimeoutError; ValueError will fail immediately
    return llm_client.enrich(text)
```

### Exception filter semantics

- **Caught exceptions:** the decorator catches only the specified exception types. All others propagate immediately.
- **Rationale:** distinguishes transient errors (network, timeout, rate limit) from permanent ones (invalid input, bad API key).
- **Example filters:**
  - `exceptions=(TimeoutError, ConnectionError)` — network transients only
  - `exceptions=(Exception,)` — catch-all (retries on any error)
  - `exceptions=()` — empty tuple (no retries, equivalent to no decorator)

### Logging

Each retry attempt logs:
- Function name
- Attempt count (e.g., "attempt 2/4")
- Exception message
- Backoff delay

After max retries, logs the final failure. This provides observability for troubleshooting slow/flaky operations.

### Integration with pipeline

The retry decorator is applied at the pipeline level to I/O-heavy operations:

| Function | Decorator | Config | Rationale |
|---|---|---|---|
| `embed_batch()` | `@retry` | max_retries=3, base=1s | Ollama transients (timeout, connection reset) |
| `upsert()` | `@retry` | max_retries=3, base=1s | ChromaDB transients (disk I/O, lock contention) |
| `enrich()` | explicit loop (DD-10) | max_retries=3, base=2s | Needs JSON validation, not just any exception |

### Notes

- **No async support:** the decorator uses `time.sleep()` (blocking). For async functions, a similar async-compatible decorator would be needed (not yet implemented).
- **Deterministic retries:** no randomization (jitter). For distributed systems, callers may add jitter externally if needed.
- **Last exception preserved:** if all retries fail, the last exception is re-raised (not a generic "max retries exceeded" error), preserving the underlying cause for debugging.

## DD-13: LLM Provider Factory Routing (`llm/factory.py`)

**Focus:** Parse "provider/model" spec → dispatch to OllamaProvider or AnthropicProvider

### `get_provider(spec, config)` algorithm

```
get_provider(spec: str, config: Config) → LLMProvider
  │
  ├─ 0. Parse spec string
  │      # Format: "provider/model_name" or "provider:model_name"
  │      # Examples: "ollama/qwen2.5-7b", "anthropic/claude-opus-4-8"
  │      
  │      if "/" in spec:
  │        provider_name, model_name = spec.split("/", 1)
  │      elif ":" in spec:
  │        provider_name, model_name = spec.split(":", 1)
  │      else:
  │        raise ValueError(f"Invalid spec format: {spec}. Expected 'provider/model'")
  │      
  │      provider_name = provider_name.lower().strip()
  │      model_name = model_name.strip()
  │
  ├─ 1. Route to provider class
  │      if provider_name == "ollama":
  │        return OllamaProvider(
  │          base_url=config.ollama_url,
  │          model=model_name,
  │          timeout=config.ollama_timeout
  │        )
  │      elif provider_name == "anthropic":
  │        return AnthropicProvider(
  │          api_key=config.anthropic_api_key,
  │          model=model_name,
  │          timeout=config.anthropic_timeout
  │        )
  │      else:
  │        raise ValueError(f"Unknown provider: {provider_name}")
  │
  └─ 2. Return provider instance
```

### Provider registry

| Provider | Base Class | Config | Notes |
|---|---|---|---|
| `ollama` | `OllamaProvider` | `ollama_url`, `ollama_timeout` | Local inference, no API key |
| `anthropic` | `AnthropicProvider` | `anthropic_api_key`, `anthropic_timeout` | Cloud API, authenticated |

### Config extraction

```python
class Config:
    # Common
    enrichment_model: str  # e.g., "ollama/qwen2.5-7b" or "anthropic/claude-opus-4-8"
    
    # Ollama
    ollama_url: str        # e.g., "http://localhost:11434"
    ollama_timeout: int    # seconds (default 30)
    
    # Anthropic
    anthropic_api_key: str # from env or config.yaml
    anthropic_timeout: int # seconds (default 60)
```

Loaded from `config.yaml` and environment variables (see DD-01).

### OllamaProvider

```python
class OllamaProvider(LLMProvider):
    def __init__(self, base_url: str, model: str, timeout: int = 30):
        self.client = ollama.Client(base_url=base_url)
        self.model = model
        self.timeout = timeout
    
    def call(self, system: str, user_message: str, json_mode: bool = False, timeout: int | None = None) → LLMResponse:
        # POST to Ollama /api/chat endpoint
        # Return LLMResponse(text=..., usage=...)
```

### AnthropicProvider

```python
class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str, model: str, timeout: int = 60):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.timeout = timeout
    
    def call(self, system: str, user_message: str, json_mode: bool = False, timeout: int | None = None) → LLMResponse:
        # Call Anthropic Messages API
        # If json_mode=True, use response_format={"type": "json_object"}
        # Return LLMResponse(text=..., usage=...)
```

### Usage

```python
# From config.yaml: enrichment_model = "ollama/qwen2.5-7b"
config = load_config("config.yaml")
provider = get_provider(config.enrichment_model, config)

response = provider.call(
    system="You are a text enrichment expert. Return JSON.",
    user_message="Enrich this text: ...",
    json_mode=True
)
```

### Error handling

| Error | Condition | Handling |
|---|---|---|
| Invalid spec format | No "/" or ":" found | Raise `ValueError` at parse time |
| Unknown provider | provider_name not in registry | Raise `ValueError` at dispatch time |
| Missing config | e.g., `ollama_url` is None | Raised by provider constructor during initialization |
| Connection error | Provider can't reach service | Handled by provider's `call()` method (may retry via DD-12) |

### Notes

- **Lazy initialization:** provider instance is created only when requested, not at config load time. This allows the service (Ollama, Anthropic) to be offline at startup.
- **Single responsibility:** the factory only routes; each provider class handles protocol details (authentication, endpoint, parsing).
- **Future extensibility:** to add a new provider (e.g., "mistral/", "openai/"), add a new `elif` branch and implement a corresponding `MistralProvider` class inheriting from `LLMProvider`.

## DD-14: Anthropic JSON Mode (`llm/anthropic_provider.py`)

**Focus:** System prompt injection + post-validation with json.loads

### `call(..., json_mode=True)` sequence

```
AnthropicProvider.call(
  system: str,
  user_message: str,
  json_mode: bool = False,
  timeout: int | None = None
) → LLMResponse
  │
  ├─ 0. Prepare system prompt
  │      if json_mode:
  │        system = system + """
  │
  │        IMPORTANT: You MUST respond with valid JSON only. 
  │        Do NOT include markdown, code blocks, explanations, or anything else.
  │        Your response must be parseable by Python's json.loads().
  │        """
  │      else:
  │        # Use system prompt as-is
  │
  ├─ 1. Call Anthropic API
  │      response = self.client.messages.create(
  │        model=self.model,
  │        max_tokens=1024,
  │        system=system,
  │        messages=[{"role": "user", "content": user_message}],
  │        timeout=timeout or self.timeout
  │      )
  │
  ├─ 2. Extract text
  │      response_text = response.content[0].text
  │
  ├─ 3. Validate JSON (if json_mode=True)
  │      if json_mode:
  │        try:
  │          parsed = json.loads(response_text)
  │          # Validation successful; return as-is
  │        except json.JSONDecodeError as e:
  │          logger.error(f"Anthropic returned invalid JSON: {e}\nText: {response_text}")
  │          raise ValueError(f"Invalid JSON response from Anthropic: {e}")
  │
  └─ 4. Return response
         LLMResponse(
           text=response_text,
           usage={
             "input_tokens": response.usage.input_tokens,
             "output_tokens": response.usage.output_tokens
           }
         )
```

### System prompt injection strategy

**Why injection:** Anthropic does not have a native `response_format={"type": "json_object"}` parameter like OpenAI does. Instead, we inject a constraint into the system prompt that strongly incentivizes JSON output.

**Prompt text:**

```
IMPORTANT: You MUST respond with valid JSON only. 
Do NOT include markdown, code blocks, explanations, or anything else.
Your response must be parseable by Python's json.loads().
```

**Placement:** appended to the user-supplied `system` prompt after a blank line separator, ensuring it doesn't conflict with domain-specific instructions.

### Validation flow

```
LLM returns response_text
  │
  ├─ json_mode = False
  │    └─ Return text as-is (no validation)
  │
  └─ json_mode = True
       │
       ├─ try json.loads(response_text)
       │    │
       │    ├─ Success: return parsed (caller validates schema)
       │    │
       │    └─ JSONDecodeError: raise ValueError (escalate as permanent error)
```

### Error handling

| Error | Cause | Handling |
|---|---|---|
| `json.JSONDecodeError` | LLM returned non-JSON (e.g., markdown with explanation) | Raise `ValueError` → caller treats as permanent failure (no retry) |
| `TimeoutError` | API call exceeded timeout | Propagate to caller → retry decorator (DD-12) retries with backoff |
| `APIError` | Anthropic API error (5xx) | Propagate to caller → retry decorator retries with backoff |
| Invalid API key | 401 Unauthorized | Propagate to caller → caller logs and skips enrichment |

### Schema validation (caller responsibility)

After `json.loads()` succeeds, the caller must validate the parsed object against the expected schema:

```python
response = provider.call(..., json_mode=True)
parsed = json.loads(response.text)

# Caller validates schema
if not isinstance(parsed, dict) or "keywords" not in parsed:
    raise ValueError("Invalid enrichment schema")
```

This separation of concerns (JSON parsing vs. schema validation) allows the provider to be generic; validation is domain-specific and lives in the enricher (DD-10).

### Token usage tracking

`usage` dict is extracted from the API response for observability:

```python
{
  "input_tokens": int,      # tokens in system + user message
  "output_tokens": int      # tokens in response
}
```

Logged or stored by callers for cost/performance monitoring.

### Notes

- **No structured output API:** unlike newer OpenAI models, Anthropic doesn't guarantee JSON output via API parameters. The system prompt injection is a pragmatic fallback.
- **LLM reliability:** Claude models are highly reliable at following JSON constraints, but edge cases (e.g., very long inputs, malformed prompts) can still produce non-JSON. The post-validation catch ensures we fail explicitly rather than silently returning garbage.
- **No streaming:** `response_format` constraints typically disable streaming to ensure atomic JSON output. This implementation doesn't use streaming, so no conflict.
- **Anthropic timeout:** set to 60s by default (longer than Ollama's 30s) because cloud API calls may have higher latency and queueing delays.

## DD-15: Search Pipeline Sequence (`search/pipeline.py`)

**Focus:** [rewrite query] → embed original + rewritten → vector search both → merge → [rerank] → top-K

### `search(query, top_k=10, rewrite=False, rerank=False)` sequence

```
search(
  query: str,
  top_k: int = 10,
  rewrite: bool = False,
  rerank: bool = False,
  collection_names: list[str] | None = None
) → list[SearchResult]
  │
  ├─ 0. Prepare search
  │      collection_names = collection_names or [all registered collections]
  │      queries_to_embed = [query]
  │
  ├─ 1. Optional query rewriting (if rewrite=True)
  │      rewritten_query = query_rewriter.rewrite(query)
  │      queries_to_embed = [query, rewritten_query]
  │      logger.info(f"Original: {query}")
  │      logger.info(f"Rewritten: {rewritten_query}")
  │
  ├─ 2. Embed all queries
  │      embeddings = []
  │      for q in queries_to_embed:
  │        emb = embedding.embed(q)
  │        embeddings.append(emb)
  │      # embeddings[0] = original embedding
  │      # embeddings[1] = rewritten embedding (if rewrite=True)
  │
  ├─ 3. Vector search each embedding
  │      all_results = []
  │      
  │      for query_text, query_embedding in zip(queries_to_embed, embeddings):
  │        results = chromadb_store.search(
  │          query_embedding,
  │          top_k=top_k,
  │          collection_names=collection_names
  │        )
  │        all_results.extend(results)
  │
  ├─ 4. Merge & deduplicate results
  │      # Group by chunk ID (or document text), merge distance info
  │      merged_results = _merge_results(all_results)
  │      # See DD-16 for merge logic
  │
  ├─ 5. Optional reranking (if rerank=True)
  │      reranked_results = reranker.rerank(
  │        query=query,
  │        results=merged_results,
  │        top_k=top_k
  │      )
  │      # See DD-17 for reranking logic
  │      merged_results = reranked_results
  │
  ├─ 6. Final top-K selection
  │      final_results = merged_results[:top_k]
  │
  └─ 7. Return results
         return final_results
```

### Configuration flags

| Flag | Default | Purpose | Performance |
|---|---|---|---|
| `rewrite` | False | Rewrite query to improve retrieval | +1 LLM call, +1 embedding, +1 vector search |
| `rerank` | False | Rerank results using LLM | +1 LLM call per result batch |
| `top_k` | 10 | Number of results to return | Fixed by caller |

### Search result merging (DD-16)

When both original and rewritten queries are embedded, results may overlap. Merging combines distance scores:

```
Original query results:      Rewritten query results:
  A (distance 0.2)             B (distance 0.15)
  B (distance 0.3)             A (distance 0.35)
  C (distance 0.5)             D (distance 0.6)

Merged (by dedup + avg distance):
  B (distance = (0.15 + 0.3) / 2 = 0.225)      ← improved by rewrite
  A (distance = (0.2 + 0.35) / 2 = 0.275)      ← original still good
  C (distance = 0.5)                            ← only in original
  D (distance = 0.6)                            ← only in rewrite
```

See DD-16 for full algorithm.

### Reranking pipeline (DD-17)

If `rerank=True`, merged results are re-scored by an LLM judge that compares the original query to each result:

```
for each result in merged_results[:top_k]:
  relevance_score = reranker.score(query, result.text)
  result.rerank_score = relevance_score

sort by rerank_score (descending)
return top_k
```

LLM judges are slower but more accurate at relevance. Trade-off: slower search for better quality.

### Integration with web API

Callers (web routes in `web/api.py`) invoke search with flags set by user preferences:

```python
@app.get("/search")
def search_api(q: str, rewrite: bool = False, rerank: bool = False, k: int = 10):
    results = search_engine.search(q, top_k=k, rewrite=rewrite, rerank=rerank)
    return {"results": [r.to_dict() for r in results]}
```

### Error handling

| Error | Handling |
|---|---|
| Query embedding fails | Propagate to caller (web route catches, returns 500) |
| Rewrite fails | Log warning, skip rewrite (fall back to original query only) |
| Rerank fails | Log warning, skip rerank (return merged results as-is) |
| Vector search fails | Propagate to caller (critical error) |

### Notes

- **Query rewriting strategy:** uses LLM to expand/clarify the user's query, improving recall when the original phrasing doesn't match the corpus. Example: "machine learning" → "ML, deep learning, neural networks, AI algorithms".
- **Merge dedup strategy:** identifies duplicate chunks (via content hash or similarity threshold) and combines their distance scores. Improves final ranking when multiple queries find the same result.
- **Reranking scope:** reranking is applied to merged results (which may be fewer than `top_k` if corpus is small), then final top-K is selected. This avoids inflating results with lower-quality rewritten hits.

## DD-16: Merge Dedup Algorithm (`search/merge.py`)

**Focus:** Group duplicate chunks → combine distance scores → sort global top-K

### `_merge_results(results: list[SearchResult])` sequence

```
_merge_results(results: list[SearchResult]) → list[SearchResult]
  │
  ├─ 0. Build dedup map
  │      dedup_key = lambda r: (r.metadata['source'], r.metadata['chunk_index'])
  │      # Or alternatively: dedup_key = lambda r: hash(r.text)
  │      
  │      chunk_groups = {}  # {dedup_key → [SearchResult, ...]}
  │      
  │      for result in results:
  │        key = dedup_key(result)
  │        if key not in chunk_groups:
  │          chunk_groups[key] = []
  │        chunk_groups[key].append(result)
  │
  ├─ 1. Merge duplicate groups
  │      merged = []
  │      
  │      for key, group in chunk_groups.items():
  │        if len(group) == 1:
  │          # No duplicate; keep as-is
  │          merged.append(group[0])
  │        else:
  │          # Multiple results for same chunk (e.g., from original + rewritten query)
  │          # Combine distance scores via averaging
  │          avg_distance = sum(r.distance for r in group) / len(group)
  │          
  │          # Keep first result's metadata, update distance
  │          merged_result = SearchResult(
  │            text=group[0].text,
  │            metadata=group[0].metadata,
  │            distance=avg_distance,
  │            collection=group[0].collection,
  │            sources=group  # preserve all source matches for transparency
  │          )
  │          merged.append(merged_result)
  │
  ├─ 2. Global sort by distance (ascending)
  │      merged.sort(key=lambda r: r.distance)
  │
  └─ 3. Return merged results (no limit; caller will top-K)
         return merged
```

### Dedup key choices

| Strategy | Key | Pros | Cons |
|---|---|---|---|
| **File + index** (recommended) | `(source, chunk_index)` | Exact match, handles chunks from same file | Requires metadata presence |
| **Content hash** | `hash(text)` | Works without metadata, catches textual duplicates | Slow on large result sets, collisions possible |
| **Exact text** | `text` | Simple, no metadata required | Case-sensitive, whitespace-sensitive |

**Recommendation:** use `(source, chunk_index)` because:
- Metadata is always present (guaranteed by ingest pipeline).
- Exact identity: same file + same chunk index = same content.
- Fast: O(n) hashing via tuple.

### Distance combination strategy

**Averaging:**
```python
avg_distance = sum(r.distance for r in group) / len(group)
```

**Rationale:**
- If original query finds a chunk at distance 0.2 and rewritten query finds it at distance 0.3, averaging gives 0.25 — penalizes the redundancy slightly while preserving strong signals.
- Alternative: take minimum (keep best match). Less conservative; risks ranking by outlier.

**Weighted average (future enhancement):**
```python
weights = [1.0, 0.8]  # rewritten query weighted lower
weighted_avg = sum(w * r.distance for w, r in zip(weights, group)) / sum(weights)
```

Allows tuning the contribution of rewritten queries (e.g., trust original more).

### SearchResult struct enhancement

To preserve merge transparency, the result includes a `sources` field:

```python
@dataclass
class SearchResult:
    text: str
    metadata: dict
    distance: float
    collection: str
    sources: list['SearchResult'] | None = None  # populated by merge
```

**Usage:** callers can inspect `sources` to see which queries matched:
```python
if result.sources and len(result.sources) > 1:
    logger.info(f"Chunk matched by {len(result.sources)} queries (improved signal)")
```

### Integration with search pipeline (DD-15)

Merge is called internally by `search()` after all vector searches complete:

```python
# After step 3 in DD-15
all_results = [SearchResult(...), SearchResult(...), ...]

# Step 4: merge dedup
merged_results = _merge_results(all_results)

# Step 5-7: continue with rerank (if enabled), then top-K
```

### Error handling

- **Empty input:** returns empty list.
- **All duplicates:** returns single merged result.
- **No duplicates:** returns original list sorted by distance.

### Notes

- **No limit in merge:** merge returns all unique chunks across all queries. The `top_k` limit is applied after reranking (DD-15 step 6).
- **Idempotent:** running merge twice on the same input produces identical results (sorts are stable).
- **Memory:** for typical searches (100-500 results), dedup map is negligible overhead. For very large result sets (>10K), consider streaming dedup.
