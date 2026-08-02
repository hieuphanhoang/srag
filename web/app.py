"""SRAG Web Application (FD-80).

FastAPI entry point that wires together:
* Service lifecycle management via ``lifespan``
* Static-file serving for the frontend
* All API routes from ``web.routes``
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from web.dependencies import get_config, lifespan
from web.routes import register_routes

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Application factory — creates and configures the FastAPI app."""
    # Configure logging first (file + console) so nothing logged during the
    # rest of startup — including this function's own log lines — is lost.
    from log import setup_logging
    cfg = get_config()
    setup_logging(level=cfg.log_level, log_file=cfg.log_file)

    # Startup is handled by ``lifespan`` in web/dependencies.py.
    app = FastAPI(
        title="SRAG",
        version="0.1.0",
        description="Semantic Retrieval-Augmented System with adaptive chunking, ChromaDB storage, and LLM enrichment.",
        lifespan=lifespan,  # type: ignore[arg-type]
    )

    # Register all API routes (FD-80).
    register_routes(app)

    # Serve static frontend files. Force revalidation on every request —
    # StaticFiles' default caching relies on browsers honoring ETag/
    # Last-Modified consistently, which isn't guaranteed, and this is a
    # small local app where correctness after an edit matters more than
    # shaving a request.
    _static_dir = Path(__file__).parent / "static"
    if _static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

        @app.middleware("http")
        async def _no_cache_static(request, call_next):
            response = await call_next(request)
            if request.url.path.startswith("/static/"):
                response.headers["Cache-Control"] = "no-cache, must-revalidate"
            return response

    # run.bat / USAGE.md point users at http://localhost:9000 directly —
    # redirect the bare root (and the commonly-guessed /ui) to the actual UI.
    @app.get("/", include_in_schema=False)
    @app.get("/ui", include_in_schema=False)
    async def _root_redirect() -> RedirectResponse:
        return RedirectResponse(url="/static/index.html")

    logger.info("SRAG FastAPI application created.")
    return app


# Module-level singleton for ``uvicorn`` discovery.
app = create_app()