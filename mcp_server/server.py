"""SRAG MCP Server (FD-78...FD-83).

Exposes SragTools over the MCP stdio transport so AI assistants (e.g. Claude
Code) can search the ingested knowledge base directly.
"""

from __future__ import annotations

import logging
from typing import Any

from mcp.server import MCPServer

from mcp_server.tools import SragTools

logger = logging.getLogger(__name__)

server = MCPServer(name="srag")


@server.tool(name="srag_search", description="Search the SRAG knowledge base for relevant document chunks.")
def srag_search(
    query: str,
    top_k: int = 10,
    rewrite_query: bool = False,
    rerank_results: bool = False,
) -> list[dict[str, Any]]:
    """Search indexed documents and return ranked chunks.

    Args:
        query: Search query text.
        top_k: Number of results to return.
        rewrite_query: Expand the query with LLM-generated variants first.
        rerank_results: Reorder merged results with an LLM relevance judge.
    """
    return SragTools.search(
        query=query,
        top_k=top_k,
        rewrite_query=rewrite_query,
        rerank_results=rerank_results,
    )


@server.tool(name="srag_ingest_file", description="Ingest a single file into the SRAG knowledge base.")
def srag_ingest_file(file_path: str) -> dict[str, Any]:
    """Queue a file's containing folder for ingestion.

    Args:
        file_path: Absolute or relative path to the file.
    """
    return SragTools.ingest_file(file_path)


@server.tool(name="srag_ingest_folder", description="Ingest every supported file in a folder into the SRAG knowledge base.")
def srag_ingest_folder(folder_path: str) -> dict[str, Any]:
    """Queue a folder for ingestion.

    Args:
        folder_path: Absolute or relative path to the folder.
    """
    return SragTools.ingest_folder(folder_path)


@server.tool(name="srag_stats", description="Get SRAG knowledge base statistics (documents, chunks, collections).")
def srag_stats() -> dict[str, Any]:
    """Return document/chunk/collection counts for the knowledge base."""
    return SragTools.get_knowledge_stats()


@server.tool(name="srag_health", description="Check SRAG storage/embedding/LLM connectivity.")
def srag_health() -> dict[str, Any]:
    """Return the health status of the storage, embedding, and LLM subsystems."""
    return SragTools.health_check()


def main() -> None:
    """Run the MCP server over stdio."""
    # File logging only - MCP's stdio transport uses stdout for the JSON-RPC
    # protocol itself, so a console log handler there would interleave log
    # lines with protocol messages and corrupt the stream.
    from config import load_config
    from log import setup_logging
    cfg = load_config()
    setup_logging(level=cfg.log_level, log_file=cfg.log_file, enable_console=False)

    server.run(transport="stdio")


if __name__ == "__main__":
    main()
