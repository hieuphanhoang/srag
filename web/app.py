"""SRAG Web Application — Phase 1 stub (DD-04).

FastAPI entry point. Currently a stub; real implementation is
a Phase 2 concern.
"""

from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(title="SRAG", version="0.1.0")


@app.get("/health")
def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/ingest")
async def ingest(file_path: str) -> dict:
    """Stub endpoint to trigger ingestion."""
    # TODO: Phase 2 - implement per DD-01, FD-46…FD-51
    return {"status": "pending", "file_path": file_path}


@app.post("/search")
async def search(query: str) -> dict:
    """Stub endpoint to trigger search."""
    # TODO: Phase 2 - implement per DD-03, FD-36…FD-45
    return {"status": "pending", "query": query}