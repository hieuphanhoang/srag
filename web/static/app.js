/** SRAG frontend — Dashboard, Local Folders, Search, Settings. */

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

function escapeHtml(str) {
  return String(str ?? "").replace(/[&<>"']/g, (m) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[m]);
}

function fmtPath(p) {
  return p ? p.replace(/\//g, "\\") : "";
}

function basename(p) {
  if (!p) return "";
  const parts = p.replace(/\\/g, "/").split("/");
  return parts[parts.length - 1] || p;
}

async function api(path, options = {}) {
  const resp = await fetch(path, {
    headers: options.body ? { "Content-Type": "application/json" } : undefined,
    ...options,
  });
  if (!resp.ok) {
    let detail = `HTTP ${resp.status}`;
    try {
      const err = await resp.json();
      detail = err.detail || detail;
    } catch { /* ignore */ }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return resp.json();
}

function folderUrl(path, suffix = "") {
  return `/api/folders/${encodeURIComponent(path)}${suffix}`;
}

// ─── Toast ──────────────────────────────────────────────────

let toastTimer = null;
function toast(message, isError = false) {
  const el = $("#toast");
  el.textContent = message;
  el.classList.toggle("error", isError);
  el.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove("show"), 3200);
}

// ─── Navigation ─────────────────────────────────────────────

const PAGES = ["dashboard", "folders", "search", "settings"];

function showPage(name) {
  if (!PAGES.includes(name)) name = "dashboard";
  $$(".page").forEach((p) => p.classList.toggle("active", p.id === `page-${name}`));
  $$(".nav-item").forEach((n) => n.classList.toggle("active", n.dataset.page === name));
  location.hash = name;
  if (name === "dashboard") refreshDashboard();
  if (name === "folders") refreshFolders();
  if (name === "settings") refreshSettings();
}

$$(".nav-item").forEach((item) => {
  item.addEventListener("click", () => showPage(item.dataset.page));
});

// ─── Top bar stats (polled everywhere) ─────────────────────

async function refreshTopbarStats() {
  try {
    const s = await api("/api/status");
    $("#statFolders").textContent = s.folders;
    $("#statChunks").textContent = s.total_chunks;
    $("#statCollections").textContent = s.collections;
    $("#connDot").className = "conn-dot " + (s.chromadb_connected ? "ok" : "bad");
    $("#connLabel").textContent = s.chromadb_connected ? "ChromaDB connected" : "ChromaDB unreachable";
    return s;
  } catch {
    $("#connDot").className = "conn-dot bad";
    $("#connLabel").textContent = "Server unreachable";
    return null;
  }
}

// ─── Dashboard ──────────────────────────────────────────────

async function refreshDashboard() {
  const s = await refreshTopbarStats();
  if (s) {
    $("#cardFolders").textContent = s.folders;
    $("#cardChunks").textContent = s.total_chunks;
    $("#cardCollections").textContent = s.collections;
    const chroma = $("#cardChroma");
    chroma.textContent = s.chromadb_connected ? "OK" : "DOWN";
    chroma.className = "value " + (s.chromadb_connected ? "status-ok" : "status-bad");
    $("#cardChromaNote").textContent = s.chromadb_connected
      ? "connection healthy" : "check server logs";
  }

  try {
    const { collections } = await api("/api/collections");
    const container = $("#collectionsTable");
    if (!collections.length) {
      container.innerHTML = '<div class="empty-state">No collections yet — register a folder to start indexing.</div>';
      return;
    }
    const maxCount = Math.max(...collections.map((c) => c.count), 1);
    container.innerHTML = collections
      .sort((a, b) => b.count - a.count)
      .map((c) => `
        <div class="folder-row" style="grid-template-columns: 1fr auto;">
          <div class="folder-main">
            <div class="folder-name">${escapeHtml(c.label)}</div>
            <div class="folder-path">${escapeHtml(c.folder_path ? fmtPath(c.folder_path) : "untracked collection · " + c.name)}</div>
            <div class="progress-track" style="margin-top:8px;">
              <div class="fill" style="width:${(c.count / maxCount * 100).toFixed(1)}%; background:var(--brass);"></div>
            </div>
          </div>
          <div class="folder-stat">${c.count} chunks</div>
        </div>
      `).join("");
  } catch (err) {
    $("#collectionsTable").innerHTML = `<div class="empty-state">Could not load collections: ${escapeHtml(err.message)}</div>`;
  }
}

// ─── Local Folders ──────────────────────────────────────────

let foldersPollTimer = null;
let expandedDetail = {}; // path -> "files" | "errors" | null

async function refreshFolders() {
  try {
    const { folders } = await api("/api/folders");
    renderFolders(folders);
    const anyActive = folders.some((f) => f.status === "indexing");
    clearTimeout(foldersPollTimer);
    if (anyActive && $("#page-folders").classList.contains("active")) {
      foldersPollTimer = setTimeout(refreshFolders, 1200);
    }
  } catch (err) {
    $("#foldersList").innerHTML = `<div class="empty-state">Could not load folders: ${escapeHtml(err.message)}</div>`;
  }
}

function renderFolders(folders) {
  const container = $("#foldersList");
  if (!folders.length) {
    container.innerHTML = '<div class="empty-state"><div class="big">No folders registered</div>Register one above to start indexing documents.</div>';
    return;
  }

  container.innerHTML = folders.map((f) => {
    const task = f.task;
    const showProgress = f.status === "indexing" && task && task.files_total > 0;
    const pct = showProgress ? Math.round(task.progress * 100) : 0;
    const detail = expandedDetail[f.path];

    return `
    <div class="folder-row" data-path="${escapeHtml(f.path)}">
      <div class="folder-main">
        <div class="folder-name">${escapeHtml(f.name)}</div>
        <div class="folder-path" title="${escapeHtml(fmtPath(f.path))}">${escapeHtml(fmtPath(f.path))}</div>
        <div class="folder-meta">
          <span class="stamp-badge ${f.status}">${f.status}</span>
          <span class="folder-stat">${f.chunk_count} chunk${f.chunk_count === 1 ? "" : "s"}</span>
          ${f.last_indexed ? `<span class="folder-stat">indexed ${timeAgo(f.last_indexed)}</span>` : ""}
          ${f.last_error ? `<span class="folder-stat" style="color:var(--danger)">${escapeHtml(f.last_error)}</span>` : ""}
        </div>
        ${showProgress ? `
          <div class="progress-track"><div class="fill" style="width:${pct}%"></div></div>
          <div class="folder-stat" style="margin-top:4px;">${task.files_done}/${task.files_total} files · ${pct}%</div>
        ` : ""}
        <div class="folder-toggles">
          <span class="toggle-item">
            <label class="switch"><input type="checkbox" class="toggle-excluded" ${f.excluded ? "checked" : ""}/><span class="track"></span></label>
            Exclude from search
          </span>
          <span class="toggle-item">
            <label class="switch"><input type="checkbox" class="toggle-single" ${f.single_search ? "checked" : ""}/><span class="track"></span></label>
            Solo (search only this folder)
          </span>
        </div>
        ${detail === "files" ? '<div class="folder-detail" data-detail="files"><div class="empty-state">Loading…</div></div>' : ""}
        ${detail === "errors" ? '<div class="folder-detail" data-detail="errors"><div class="empty-state">Loading…</div></div>' : ""}
      </div>
      <div class="folder-actions">
        <div class="btn-row">
          <button class="ghost act-ingest">Ingest</button>
          <button class="ghost act-prioritize">Prioritize</button>
          <button class="ghost act-sync">Sync</button>
        </div>
        <div class="btn-row">
          <button class="ghost act-files">Files</button>
          <button class="ghost act-errors">Errors</button>
          <button class="ghost danger act-remove">Remove</button>
        </div>
      </div>
    </div>`;
  }).join("");

  // Wire up per-row actions.
  $$(".folder-row", container).forEach((row) => {
    const path = row.dataset.path;

    $(".act-ingest", row).addEventListener("click", () => runFolderAction(path, "ingest", "Queued for ingest."));
    $(".act-prioritize", row).addEventListener("click", () => runFolderAction(path, "prioritize", "Moved to front of the queue."));
    $(".act-sync", row).addEventListener("click", () => runFolderAction(path, "sync", "Queued for sync."));
    $(".act-remove", row).addEventListener("click", () => removeFolder(path));

    $(".act-files", row).addEventListener("click", () => toggleDetail(path, "files"));
    $(".act-errors", row).addEventListener("click", () => toggleDetail(path, "errors"));

    $(".toggle-excluded", row).addEventListener("change", async (e) => {
      try { await api(folderUrl(path, "/toggle-excluded"), { method: "PATCH" }); }
      catch (err) { toast(err.message, true); e.target.checked = !e.target.checked; }
    });
    $(".toggle-single", row).addEventListener("change", async (e) => {
      try { await api(folderUrl(path, "/toggle-single"), { method: "PATCH" }); refreshFolders(); }
      catch (err) { toast(err.message, true); e.target.checked = !e.target.checked; }
    });
  });

  // Populate any open detail panels.
  Object.entries(expandedDetail).forEach(([path, kind]) => {
    if (kind) loadDetail(path, kind);
  });
}

async function runFolderAction(path, action, message) {
  try {
    await api(folderUrl(path, `/${action}`), { method: "POST" });
    toast(message);
    refreshFolders();
  } catch (err) {
    toast(err.message, true);
  }
}

async function removeFolder(path) {
  if (!confirm(`Remove "${basename(path)}" and delete its indexed data? This cannot be undone.`)) return;
  try {
    await api(folderUrl(path), { method: "DELETE" });
    toast("Folder removed.");
    delete expandedDetail[path];
    refreshFolders();
  } catch (err) {
    toast(err.message, true);
  }
}

function toggleDetail(path, kind) {
  expandedDetail[path] = expandedDetail[path] === kind ? null : kind;
  refreshFolders();
}

async function loadDetail(path, kind) {
  const row = $(`.folder-row[data-path="${cssEscape(path)}"]`);
  if (!row) return;
  const el = $(`.folder-detail[data-detail="${kind}"]`, row);
  if (!el) return;

  try {
    if (kind === "files") {
      const { files } = await api(folderUrl(path, "/files"));
      el.innerHTML = files.length
        ? files.map((f) => `<div class="file-row"><span>${escapeHtml(basename(f.path))}</span><span>${(f.size / 1024).toFixed(1)} KB</span></div>`).join("")
        : '<div class="empty-state">No supported files found.</div>';
    } else {
      const { errors } = await api(folderUrl(path, "/errors"));
      el.innerHTML = errors.length
        ? errors.map((e) => `<div class="err-row">${escapeHtml(e)}</div>`).join("")
        : '<div class="empty-state">No errors on the last run.</div>';
    }
  } catch (err) {
    el.innerHTML = `<div class="err-row">${escapeHtml(err.message)}</div>`;
  }
}

function cssEscape(s) {
  return s.replace(/["\\]/g, "\\$&");
}

function timeAgo(ts) {
  const sec = Math.max(0, Date.now() / 1000 - ts);
  if (sec < 60) return "just now";
  if (sec < 3600) return `${Math.floor(sec / 60)}m ago`;
  if (sec < 86400) return `${Math.floor(sec / 3600)}h ago`;
  return `${Math.floor(sec / 86400)}d ago`;
}

$("#registerForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const input = $("#folderPathInput");
  const path = input.value.trim();
  if (!path) return;

  const btn = $("#registerBtn");
  const errEl = $("#registerError");
  errEl.style.display = "none";
  btn.disabled = true;

  try {
    await api("/api/folders", { method: "POST", body: JSON.stringify({ path }) });
    input.value = "";
    toast("Folder registered — ingest started.");
    refreshFolders();
  } catch (err) {
    errEl.textContent = err.message;
    errEl.style.display = "block";
  } finally {
    btn.disabled = false;
  }
});

// ─── Search ─────────────────────────────────────────────────

$("#searchForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const query = $("#queryInput").value.trim();
  if (!query) return;

  const btn = $("#searchBtn");
  const results = $("#searchResults");
  btn.disabled = true;
  results.innerHTML = '<div class="empty-state">Searching…</div>';

  try {
    const body = {
      query,
      top_k: parseInt($("#topKInput").value, 10) || 10,
      rewrite: $("#rewriteCheck").checked,
      rerank: $("#rerankCheck").checked,
    };
    const data = await api("/api/search", { method: "POST", body: JSON.stringify(body) });
    renderResults(data);
  } catch (err) {
    results.innerHTML = `<div class="empty-state"><div class="big">Search failed</div>${escapeHtml(err.message)}</div>`;
  } finally {
    btn.disabled = false;
  }
});

function renderResults(data) {
  const container = $("#searchResults");
  if (!data.results.length) {
    container.innerHTML = '<div class="empty-state"><div class="big">No matches</div>Try a shorter or more general phrase, or enable query rewriting.</div>';
    return;
  }

  const rewriteNote = data.rewritten_query
    ? `<div class="folder-stat" style="margin-bottom:10px;">Also searched as: <span class="mono">${escapeHtml(data.rewritten_query)}</span></div>`
    : "";

  container.innerHTML = rewriteNote + data.results.map((r, i) => {
    const pct = Math.round(r.score * 100);
    return `
    <div class="result-item">
      <div class="result-rank mono">${i + 1}</div>
      <div class="result-body">
        <div class="result-header">
          <span class="result-title">${escapeHtml(r.filename || "Untitled")}</span>
          <span class="result-collection">${escapeHtml(r.collection || "—")}</span>
        </div>
        <div class="result-source" title="${escapeHtml(fmtPath(r.source))}">${escapeHtml(fmtPath(r.source))}${r.chunk_index != null ? ` · chunk ${r.chunk_index}${r.total_chunks ? "/" + r.total_chunks : ""}` : ""}</div>
        <div class="result-text">${escapeHtml(r.text)}</div>
        <span class="result-expand">Show more</span>
        <div class="ruler">
          <div class="ruler-track">
            <div class="ruler-fill" style="width:${pct}%"></div>
            <div class="ruler-notch" style="left:${pct}%"></div>
          </div>
          <span class="ruler-reading">${pct}% · d=${r.distance.toFixed(3)}${r.rerank_score != null ? ` · rr=${r.rerank_score.toFixed(2)}` : ""}</span>
        </div>
      </div>
    </div>`;
  }).join("");

  $$(".result-expand", container).forEach((btn) => {
    btn.addEventListener("click", () => {
      const textEl = btn.previousElementSibling;
      const expanded = textEl.classList.toggle("expanded");
      btn.textContent = expanded ? "Show less" : "Show more";
    });
  });
}

// ─── Settings ───────────────────────────────────────────────

async function refreshSettings() {
  try {
    const cfg = await api("/api/config");
    $("#configList").innerHTML = `
      <dt>ChromaDB path</dt><dd>${escapeHtml(fmtPath(cfg.chromadb_path))}</dd>
      <dt>Ollama URL</dt><dd>${escapeHtml(cfg.ollama_url)}</dd>
      <dt>Embedding model</dt><dd>${escapeHtml(cfg.embedding_model)}</dd>
      <dt>Server port</dt><dd>${cfg.server_port}</dd>
      <dt>Chunking</dt><dd>${cfg.chunking_max_tokens} tokens · ${cfg.chunking_overlap} overlap</dd>
      <dt>Rewrite model</dt><dd>${escapeHtml(cfg.llm_rewrite)}</dd>
      <dt>Rerank model</dt><dd>${escapeHtml(cfg.llm_rerank)}</dd>
      <dt>Enrichment model</dt><dd>${escapeHtml(cfg.llm_enrichment)}</dd>
    `;
    $("#cfgRewrite").checked = cfg.search_rewrite_enabled;
    $("#cfgRerank").checked = cfg.search_rerank_enabled;
    $("#cfgFinalK").value = cfg.search_final_k;
  } catch (err) {
    $("#configList").innerHTML = `<dt>Error</dt><dd>${escapeHtml(err.message)}</dd>`;
  }
}

async function patchConfig(updates) {
  try {
    await api("/api/config", { method: "PATCH", body: JSON.stringify(updates) });
    toast("Setting updated (runtime only).");
  } catch (err) {
    toast(err.message, true);
  }
}

$("#cfgRewrite").addEventListener("change", (e) => patchConfig({ search_rewrite_enabled: e.target.checked }));
$("#cfgRerank").addEventListener("change", (e) => patchConfig({ search_rerank_enabled: e.target.checked }));
$("#cfgFinalK").addEventListener("change", (e) => {
  const v = parseInt(e.target.value, 10);
  if (v > 0) patchConfig({ search_final_k: v });
});

// ─── Init ───────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
  const initial = location.hash.replace("#", "");
  showPage(PAGES.includes(initial) ? initial : "dashboard");
  refreshTopbarStats();
  setInterval(refreshTopbarStats, 8000);
});
