#!/usr/bin/env bash
# SRAG Phase 1 startup script (FD-85).
# Launches the SRAG application — web server, MCP server, and optional eval runner.
# Usage:
#   ./run.sh              # Run all services
#   ./run.sh web          # Run only the web server
#   ./run.sh mcp          # Run only the MCP server
#   ./run.sh eval         # Run evaluation (requires test dataset)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- venv bootstrap ----------------------------------------------------------
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

source ".venv/bin/activate"

echo "Installing dependencies (first run only)..."
pip install -q -e . 2>/dev/null || pip install -q .

# --- config -------------------------------------------------------------------
export SRAG_CONFIG="${SRAG_CONFIG:-./config.yaml}"
export SRAG_LOG_FILE="${SRAG_LOG_FILE:-./logs/srag.log}"

# --- helper ------------------------------------------------------------------
run_web() {
    echo "=== Starting SRAG web server on :9001 ==="
    uvicorn "web.app:create_app" \
        "--app-dir" "." \
        --factory \
        --host 0.0.0.0 \
        --port 9001 \
        --log-config "$(dirname "$0")/log_config.py"
}

run_mcp() {
    echo "=== Starting SRAG MCP server ==="
    python -m mcp_server.server
}

run_eval() {
    echo "=== Running retrieval evaluation ==="
    echo "Requires eval/test_dataset.jsonl and a running Ollama instance."
    python -c "from eval import runner; print('Evaluation runner stub')"
}

# --- main --------------------------------------------------------------------
case "${1:-all}" in
    web)
        run_web
        ;;
    mcp)
        run_mcp
        ;;
    eval)
        run_eval
        ;;
    all|"")
        echo "=== SRAG Phase 1 — Skeleton + Logging ==="
        echo "Config : $SRAG_CONFIG"
        echo "Log    : $SRAG_LOG_FILE"
        echo "Web UI : http://localhost:9001/ui"
        run_web
        ;;
    *)
        echo "Usage: $0 {all|web|mcp|eval}" >&2
        exit 1
        ;;
esac