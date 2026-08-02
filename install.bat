@echo off
REM SRAG Installer — one-time environment setup for a fresh Windows machine.
REM After this completes, use run.bat to start the app (no further setup needed).
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ============================================================
echo  SRAG Installer
echo ============================================================
echo.

REM ------------------------------------------------------------------
REM 1. Check / install uv (manages its own Python toolchain per
REM    pyproject.toml's requires-python - no separate Python check needed)
REM ------------------------------------------------------------------
echo [1/5] Checking uv package manager...
where uv >nul 2>&1
if errorlevel 1 (
    echo   uv not found - installing via the official installer...
    powershell -NoProfile -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    if errorlevel 1 (
        echo ERROR: uv installation failed. Install manually:
        echo   https://docs.astral.sh/uv/getting-started/installation/
        goto :fail
    )
    REM The installer updates the persistent user PATH, but this already-
    REM running process doesn't see that until a new shell starts - extend
    REM this session's PATH so the rest of this script can use uv immediately.
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
    where uv >nul 2>&1
    if errorlevel 1 (
        echo ERROR: uv installed but not found on PATH in this session.
        echo   Close this window, open a new terminal, and re-run install.bat.
        goto :fail
    )
)
for /f "tokens=*" %%V in ('uv --version 2^>^&1') do echo   Found %%V
echo.

REM ------------------------------------------------------------------
REM 2. Install Python dependencies into .venv
REM ------------------------------------------------------------------
echo [2/5] Installing dependencies (uv sync)...
uv sync
if errorlevel 1 (
    echo ERROR: uv sync failed - see output above.
    goto :fail
)
echo   Dependencies installed into .venv
echo.

REM ------------------------------------------------------------------
REM 3. Sanity-check the install
REM ------------------------------------------------------------------
echo [3/5] Verifying the install...
uv run python -c "import config, log, models; from config import load_config; load_config()" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Core modules failed to import - see below.
    uv run python -c "import config, log, models; from config import load_config; load_config()"
    goto :fail
)
echo   Core modules import cleanly.
echo.

REM ------------------------------------------------------------------
REM 4. Ollama + the embedding model (required - search cannot work
REM    without it). LLM models for rewrite/rerank/enrichment are
REM    optional (those stages degrade gracefully when unavailable) -
REM    notify only, don't auto-pull, since config.yaml's exact model
REM    choice there is a preference call, not a hard requirement.
REM ------------------------------------------------------------------
echo [4/5] Checking Ollama...
where ollama >nul 2>&1
if errorlevel 1 (
    echo   WARNING: Ollama not found on PATH.
    echo   SRAG requires a running Ollama instance for embeddings.
    echo   Install it from https://ollama.com/download, then run:
    echo     ollama pull qwen3-embedding:8b
    goto :ollama_done
)

curl -s -o nul http://localhost:11434/api/version
if errorlevel 1 (
    echo   WARNING: Ollama is installed but doesn't appear to be running.
    echo   Start Ollama, then run:  ollama pull qwen3-embedding:8b
    goto :ollama_done
)
echo   Ollama is running.

ollama list | findstr /I "qwen3-embedding-8b" >nul 2>&1
if not errorlevel 1 (
    echo   Embedding model "qwen3-embedding-8b" already present.
    goto :ollama_done
)

echo   Pulling qwen3-embedding:8b (multi-GB download, may take a while)...
ollama pull qwen3-embedding:8b
if errorlevel 1 (
    echo   WARNING: Pull failed. Once network/Ollama issues are resolved, run:
    echo     ollama pull qwen3-embedding:8b
    goto :ollama_done
)

echo   Aliasing it to "qwen3-embedding-8b" to match config.yaml...
ollama cp qwen3-embedding:8b qwen3-embedding-8b >nul 2>&1
if errorlevel 1 (
    echo   WARNING: Alias failed - either edit config.yaml's embedding_model
    echo   to "qwen3-embedding:8b", or run manually:
    echo     ollama cp qwen3-embedding:8b qwen3-embedding-8b
) else (
    echo   Done.
)

:ollama_done
echo.
echo   config.yaml's optional LLM stages (query rewrite / rerank / enrichment)
echo   are currently set to:
findstr /R "^  enrichment:\|^  rewrite:\|^  rerank:\|^  eval:" config.yaml
echo   These are optional - SRAG works fully without them; missing models
echo   just make those specific stages a no-op. To enable them, pull a
echo   model (e.g. "ollama pull qwen3:8b") and point config.yaml's llm
echo   section at it (e.g. "ollama/qwen3:8b").
echo.

REM ------------------------------------------------------------------
REM 5. Register the MCP server with Claude Code (best-effort, optional)
REM ------------------------------------------------------------------
echo [5/5] Registering the MCP server with Claude Code...
where claude >nul 2>&1
if errorlevel 1 (
    echo   Claude Code CLI not found - skipping.
    echo   To register it later once Claude Code is installed, run:
    echo     claude mcp add srag --scope user -- uv run --directory "%CD%" python -m mcp_server.server
    goto :mcp_done
)
claude mcp add srag --scope user -- uv run --directory "%CD%" python -m mcp_server.server
if errorlevel 1 (
    echo   WARNING: Registration failed ^(may already be registered^). To retry:
    echo     claude mcp add srag --scope user -- uv run --directory "%CD%" python -m mcp_server.server
) else (
    echo   Registered "srag" as an MCP server at user scope.
)

:mcp_done
echo.
echo ============================================================
echo  Install complete.
echo  Run run.bat to start the web UI at http://localhost:9000
echo ============================================================
pause
exit /b 0

:fail
echo.
echo Install failed - see the error above.
pause
exit /b 1
