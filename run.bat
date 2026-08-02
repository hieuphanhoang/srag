@echo off
REM SRAG Run Script — Web UI / MCP server entry point

if "%~1"=="" goto :web
goto :mcp

:web
call :free_port 9000
echo Starting web UI at http://localhost:9000...
REM uv run resolves this project's own .venv regardless of what's on PATH -
REM a bare "python" call here would depend on the venv already being
REM activated or otherwise being first on PATH, which install.bat does not
REM guarantee on a machine that has never activated it in that shell.
uv run python -m uvicorn web.app:create_app --factory --host 0.0.0.0 --port 9000 --reload
goto :end

:mcp
echo Starting MCP server...
uv run python -m mcp_server.server
goto :end

:free_port
REM Kill whatever is already listening on this port so re-running this
REM script behaves like a restart instead of failing with:
REM   ERROR: [WinError 10013] An attempt was made to access a socket in a
REM   way forbidden by its access permissions
setlocal
set "PORT=%~1"
set "FOUND="
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /R /C:":%PORT% .*LISTENING"') do (
    if not "%%P"=="%FOUND%" (
        echo Port %PORT% is already in use by PID %%P - stopping it...
        REM /T kills the whole process tree. --reload spawns its actual
        REM worker as a separate multiprocessing child process; killing
        REM just the reloader leaves that child orphaned and still holding
        REM the port, so a plain "taskkill /F /PID" here is not enough.
        taskkill /F /T /PID %%P >nul 2>&1
        set "FOUND=%%P"
    )
)
endlocal
goto :eof

:end
pause