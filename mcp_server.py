#!/usr/bin/env python3
"""
paper-rag MCP server — expose your local, fully-offline paper corpus as MCP tools.

Any MCP client (Claude Desktop, Claude Code, Cursor, ...) can then call:
  search_papers(query, k) — retrieve the most relevant passages (hybrid + rerank) with
                            source/page citations, WITHOUT an LLM — let the client's model reason.
  ask_papers(question)    — retrieve + answer with the *local* LLM, inline [n] citations.
  corpus_stats()          — how many chunks are indexed, and in which mode.

Everything runs locally. Configure retrieval/LLM with the same env vars as the CLI
(OLLAMA_URL, RAG_LLM, RAG_EMBED, RAG_DB, RAG_RERANK, RAG_NUM_CTX, ...) — e.g. point
OLLAMA_URL at a GPU box and RAG_LLM at a bigger model.

Run (stdio transport — the default for desktop MCP clients):
  pip install mcp pypdf qdrant-client fastembed
  python rag.py ingest ./papers      # ingest first (CLI), then start the server
  python mcp_server.py

Client config (Claude Desktop claude_desktop_config.json, or an .mcp.json):
  {
    "mcpServers": {
      "paper-rag": {
        "command": "python",
        "args": ["/abs/path/to/mcp_server.py"],
        "env": {
          "RAG_DB": "/abs/path/to/rag_qdrant",
          "OLLAMA_URL": "http://127.0.0.1:11434",
          "RAG_LLM": "qwen3:8b"
        }
      }
    }
  }

Note: the embedded Qdrant DB is single-process — ingest with the server stopped, then serve.
"""
from mcp.server.fastmcp import FastMCP
import rag

mcp = FastMCP("paper-rag")


@mcp.tool()
def search_papers(query: str, k: int = 5) -> list[dict]:
    """Search the local paper corpus and return the most relevant passages.

    Hybrid retrieval (dense BGE-M3 + BM25 sparse, RRF-fused) followed by a cross-encoder
    reranker. No LLM is involved — the caller's own model can reason over the passages.
    Each item: {text, source, page, score}.
    """
    return rag.retrieve(query, k=k)


@mcp.tool()
def ask_papers(question: str) -> dict:
    """Answer a question over the local paper corpus using the local LLM, with inline [n] citations.

    Returns {answer, sources}. Fully offline — nothing leaves the machine.
    """
    return rag.answer(question)


@mcp.tool()
def corpus_stats() -> dict:
    """How many chunks are indexed, where, the retrieval mode (hybrid vs dense-only), and the LLM in use."""
    info = rag.client().get_collection(rag.COLLECTION)
    return {"chunks": info.points_count, "db": rag.QDRANT_PATH,
            "mode": "hybrid" if rag.HYBRID else "dense-only", "llm": rag.LLM_MODEL}


if __name__ == "__main__":
    mcp.run()   # stdio transport
