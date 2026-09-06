from datetime import datetime, timezone
import logging
from typing import Any
try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    class FastMCP:  # type: ignore
        def __init__(self, name: str):
            self.name = name

        def tool(self):
            def decorator(fn):
                return fn
            return decorator

        def run(self, transport="stdio"):
            pass


from src.agents.graph import get_langfuse_client
from src.retrieval.hybrid import _GLOBAL_DOCUMENT_STORE, hybrid_retrieve

logger = logging.getLogger(__name__)

mcp = FastMCP("MuhawirKnowledgeBase")


def _log_tool_to_langfuse(tool_name: str, args: dict[str, Any], result: Any) -> None:
    client = get_langfuse_client()
    if client:
        try:
            client.trace(name=f"mcp_tool:{tool_name}", input=args, output=result)
        except Exception as e:
            logger.debug("Langfuse trace failed for tool %s: %s", tool_name, e)


@mcp.tool()
async def search_documents(
    query: str,
    language: str = "auto",
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Search documents in the knowledge base using BM25 and vector search with RRF."""
    if not query or not query.strip():
        return [{"error": "Invalid query: Query string cannot be empty."}]

    try:
        docs = await hybrid_retrieve(query=query.strip(), language=language, top_k=top_k)
        results = [
            {
                "doc_id": d["doc_id"],
                "title": d.get("metadata", {}).get("title", d["source"]),
                "snippet": d["text"][:300] if d.get("text") else "",
                "score": round(d["score"], 4),
            }
            for d in docs
        ]
        _log_tool_to_langfuse(
            "search_documents",
            {"query": query, "language": language, "top_k": top_k},
            results,
        )
        return results
    except Exception as e:
        logger.error("search_documents tool execution error: %s", e)
        return [{"error": f"Search failed: {str(e)}"}]


@mcp.tool()
async def get_document_content(doc_id: str) -> dict[str, Any]:
    """Retrieve full text and metadata for a specific document by doc_id."""
    if not doc_id:
        return {"error": "Invalid doc_id: Must provide a valid document ID."}

    doc = _GLOBAL_DOCUMENT_STORE.get(doc_id)
    if not doc:
        # Search chunks if doc_id might refer to a chunk or base doc
        for k, v in _GLOBAL_DOCUMENT_STORE.items():
            if v.get("doc_id") == doc_id or k == doc_id:
                doc = v
                break

    if not doc:
        return {"error": f"Document with id '{doc_id}' not found."}

    result = {
        "doc_id": doc.get("doc_id", doc_id),
        "title": doc.get("title", doc.get("metadata", {}).get("title", "")),
        "full_text": doc.get("text", ""),
        "metadata": doc.get("metadata", {}),
    }
    _log_tool_to_langfuse("get_document_content", {"doc_id": doc_id}, result)
    return result


@mcp.tool()
async def list_sources() -> dict[str, Any]:
    """List document count and summary breakdown by language."""
    ar_count = 0
    en_count = 0
    seen_docs = set()

    for doc in _GLOBAL_DOCUMENT_STORE.values():
        did = doc.get("doc_id", "")
        if did in seen_docs:
            continue
        seen_docs.add(did)
        lang = doc.get("language", "en")
        if lang == "ar":
            ar_count += 1
        else:
            en_count += 1

    result = {
        "total_docs": len(seen_docs),
        "by_language": {
            "ar": ar_count,
            "en": en_count,
        },
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }
    _log_tool_to_langfuse("list_sources", {}, result)
    return result


if __name__ == "__main__":
    mcp.run(transport="stdio")
