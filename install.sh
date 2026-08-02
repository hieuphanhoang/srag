#!/usr/bin/env bash
# SRAG Installer — one-time environment setup for a fresh macOS/Linux machine.
# After this completes, use run.sh to start the app (no further setup needed).

set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================"
echo " SRAG Installer"
echo "============================================================"
echo

fail() {
    echo
    echo "Install failed - see the error above."
    exit 1
}

# ------------------------------------------------------------------
# 1. Check / install uv (manages its own Python toolchain per
#    pyproject.toml's requires-python - no separate Python check needed)
# ------------------------------------------------------------------
echo "[1/5] Checking uv package manager..."
if ! command -v uv >/dev/null 2>&1; then
    echo "  uv not found - installing via the official installer..."
    if ! curl -LsSf https://astral.sh/uv/install.sh | sh; then
        echo "ERROR: uv installation failed. Install manually:"
        echo "  https://docs.astral.sh/uv/getting-started/installation/"
        fail
    fi
    # The installer updates shell profile files for future sessions, but
    # this already-running script doesn't see that until a new shell
    # starts - extend this session's PATH so the rest of the script can
    # use uv immediately.
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    if ! command -v uv >/dev/null 2>&1; then
        echo "ERROR: uv installed but not found on PATH in this session."
        echo "  Open a new terminal and re-run install.sh."
        fail
    fi
fi
echo "  Found $(uv --version)"
echo

# ------------------------------------------------------------------
# 2. Install Python dependencies into .venv
# ------------------------------------------------------------------
echo "[2/5] Installing dependencies (uv sync)..."
if ! uv sync; then
    echo "ERROR: uv sync failed - see output above."
    fail
fi
echo "  Dependencies installed into .venv"
echo

# ------------------------------------------------------------------
# 3. Sanity-check the install
# ------------------------------------------------------------------
echo "[3/5] Verifying the install..."
if ! uv run python -c "import config, log, models; from config import load_config; load_config()" >/dev/null 2>&1; then
    echo "ERROR: Core modules failed to import - see below."
    uv run python -c "import config, log, models; from config import load_config; load_config()"
    fail
fi
echo "  Core modules import cleanly."
echo

# ------------------------------------------------------------------
# 4. Ollama + the embedding model (required - search cannot work
#    without it). LLM models for rewrite/rerank/enrichment are
#    optional (those stages degrade gracefully when unavailable) -
#    notify only, don't auto-pull, since config.yaml's exact model
#    choice there is a preference call, not a hard requirement.
# ------------------------------------------------------------------
echo "[4/5] Checking Ollama..."
if ! command -v ollama >/dev/null 2>&1; then
    echo "  WARNING: Ollama not found on PATH."
    echo "  SRAG requires a running Ollama instance for embeddings."
    echo "  Install it from https://ollama.com/download, then run:"
    echo "    ollama pull qwen3-embedding:8b"
elif ! curl -s -o /dev/null http://localhost:11434/api/version; then
    echo "  WARNING: Ollama is installed but doesn't appear to be running."
    echo "  Start Ollama, then run:  ollama pull qwen3-embedding:8b"
else
    echo "  Ollama is running."
    if ollama list | grep -qi "qwen3-embedding-8b"; then
        echo "  Embedding model \"qwen3-embedding-8b\" already present."
    else
        echo "  Pulling qwen3-embedding:8b (multi-GB download, may take a while)..."
        if ollama pull qwen3-embedding:8b; then
            echo "  Aliasing it to \"qwen3-embedding-8b\" to match config.yaml..."
            if ollama cp qwen3-embedding:8b qwen3-embedding-8b >/dev/null 2>&1; then
                echo "  Done."
            else
                echo "  WARNING: Alias failed - either edit config.yaml's embedding_model"
                echo "  to \"qwen3-embedding:8b\", or run manually:"
                echo "    ollama cp qwen3-embedding:8b qwen3-embedding-8b"
            fi
        else
            echo "  WARNING: Pull failed. Once network/Ollama issues are resolved, run:"
            echo "    ollama pull qwen3-embedding:8b"
        fi
    fi
fi
echo
echo "  config.yaml's optional LLM stages (query rewrite / rerank / enrichment)"
echo "  are currently set to:"
grep -E "^  (enrichment|rewrite|rerank|eval):" config.yaml || true
echo "  These are optional - SRAG works fully without them; missing models"
echo "  just make those specific stages a no-op. To enable them, pull a"
echo "  model (e.g. \"ollama pull qwen3:8b\") and point config.yaml's llm"
echo "  section at it (e.g. \"ollama/qwen3:8b\")."
echo

# ------------------------------------------------------------------
# 5. Register the MCP server with Claude Code (best-effort, optional)
# ------------------------------------------------------------------
echo "[5/5] Registering the MCP server with Claude Code..."
INSTALL_DIR="$(pwd)"
if ! command -v claude >/dev/null 2>&1; then
    echo "  Claude Code CLI not found - skipping."
    echo "  To register it later once Claude Code is installed, run:"
    echo "    claude mcp add srag --scope user -- uv run --directory \"$INSTALL_DIR\" python -m mcp_server.server"
else
    if claude mcp add srag --scope user -- uv run --directory "$INSTALL_DIR" python -m mcp_server.server; then
        echo "  Registered \"srag\" as an MCP server at user scope."
    else
        echo "  WARNING: Registration failed (may already be registered). To retry:"
        echo "    claude mcp add srag --scope user -- uv run --directory \"$INSTALL_DIR\" python -m mcp_server.server"
    fi
fi

echo
echo "============================================================"
echo " Install complete."
echo " Run ./run.sh to start the web UI at http://localhost:9000"
echo "============================================================"
