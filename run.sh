#!/usr/bin/env bash
# SRAG Run Script — Web UI / MCP server entry point (macOS/Linux).
# Mirrors run.bat's behavior. Run install.sh once first to set up .venv.
#
# Usage:
#   ./run.sh          # Web UI on http://localhost:9000
#   ./run.sh mcp       # MCP server over stdio

set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

free_port() {
    local port="$1"
    local pids
    pids="$(lsof -ti tcp:"$port" 2>/dev/null || true)"
    if [ -n "$pids" ]; then
        echo "Port $port is already in use by PID(s) $pids - stopping them..."
        # Kill each listener's whole process group, not just the listed PID:
        # uvicorn --reload spawns its actual worker as a separate child
        # process (see run.bat's equivalent /T note) - killing only the
        # reloader would leave that child orphaned and still holding the port.
        for pid in $pids; do
            pkill -9 -P "$pid" 2>/dev/null || true
            kill -9 "$pid" 2>/dev/null || true
        done
    fi
}

case "${1:-web}" in
    web|"")
        free_port 9000
        echo "Starting web UI at http://localhost:9000..."
        # uv run resolves this project's own .venv regardless of what's on
        # PATH - matches run.bat's use of "uv run python" over a bare
        # "python" call, which would depend on the venv already being
        # activated in this shell.
        exec uv run python -m uvicorn web.app:create_app --factory --host 0.0.0.0 --port 9000 --reload
        ;;
    mcp)
        echo "Starting MCP server..."
        exec uv run python -m mcp_server.server
        ;;
    *)
        echo "Usage: $0 [web|mcp]" >&2
        exit 1
        ;;
esac
