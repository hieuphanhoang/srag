@echo off
REM SRAG Run Script (Windows) — Phase 1 ready
REM Usage: run.bat [mcp|web]

set PYTHON=%PYTHON:=%
if "%PYTHON%"=="" set PYTHON=python

REM Check if virtual environment exists
if exist "venv\Scripts\python.exe" (
    set PYTHON=venv\Scripts\python.exe
)

REM Install dependencies if needed
echo Checking dependencies...
%PYTHON% -m pip install -e ".[dev]" --quiet 2>nul
if errorlevel 1 (
    echo Failed to install dependencies. Please install manually.
    goto :error
)

REM Run the requested mode
if "%~1"=="" (
    echo Usage: run.bat [mcp|web]
    goto :error
)

if "%~1"=="mcp" (
    echo Starting MCP server...
    %PYTHON% -m mcp_server
) else if "%~1"=="web" (
    echo Starting web server on http://localhost:8000...
    %PYTHON% -m uvicorn web.app:create_app --factory --host 0.0.0.0 --port 8000 --reload
) else (
    echo Unknown mode: %~1
    echo Usage: run.bat [mcp|web]
    goto :error
)

goto :end

:error
echo Error: Failed to start server.
exit /b 1

:end
echo Done.