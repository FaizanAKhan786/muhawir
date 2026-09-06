import functools
import logging
from typing import Any, Callable, Literal
from langgraph.graph import END, StateGraph

from src.agents.nodes import (
    detect_language,
    expand_query,
    format_response,
    generate,
    quality_gate,
    rerank,
    retrieve,
)
from src.agents.state import AgentState
from src.config import get_settings

logger = logging.getLogger(__name__)

# Optional Langfuse client
_LANGFUSE_CLIENT: Any = None


def get_langfuse_client():
    global _LANGFUSE_CLIENT
    if _LANGFUSE_CLIENT is None:
        settings = get_settings()
        if settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY:
            try:
                from langfuse import Langfuse

                _LANGFUSE_CLIENT = Langfuse(
                    public_key=settings.LANGFUSE_PUBLIC_KEY,
                    secret_key=settings.LANGFUSE_SECRET_KEY,
                    host=settings.LANGFUSE_HOST,
                )
            except Exception as e:
                logger.warning("Could not initialize Langfuse: %s", e)
    return _LANGFUSE_CLIENT


def trace_node(name: str, fn: Callable):
    """Wrap agent node execution with optional Langfuse tracing span."""

    @functools.wraps(fn)
    async def wrapped(state: AgentState) -> dict[str, Any]:
        client = get_langfuse_client()
        trace_id = state.get("trace_id", "")
        span = None

        if client and trace_id:
            try:
                span = client.span(
                    name=f"agent_node:{name}",
                    trace_id=trace_id,
                    input={"query": state.get("query"), "retry_count": state.get("retry_count", 0)},
                )
            except Exception as e:
                logger.debug("Langfuse span creation skipped: %s", e)

        try:
            result = await fn(state)
            if span:
                span.end(output=result)
            return result
        except Exception as err:
            if span:
                span.end(level="ERROR", status_message=str(err))
            raise

    return wrapped


def check_quality_gate(state: AgentState) -> Literal["format_response", "expand_query"]:
    """Conditional edge decision after quality_gate.

    - PASS → format_response
    - FAIL and retry_count < MAX_RETRIES → expand_query
    - FAIL and retry_count >= MAX_RETRIES → format_response
    """
    settings = get_settings()
    quality = state.get("quality_score", {"faithfulness": 1.0, "language_consistent": True, "grounded": True})
    retry_count = state.get("retry_count", 0)

    is_faithful = quality.get("faithfulness", 0.0) >= settings.FAITHFULNESS_THRESHOLD
    is_lang_ok = quality.get("language_consistent", True)
    passed = is_faithful and is_lang_ok

    if passed:
        logger.info("Quality gate PASSED (faithfulness=%.2f, lang_ok=%s)", quality.get("faithfulness", 0.0), is_lang_ok)
        return "format_response"

    if retry_count < settings.MAX_RETRIES:
        logger.info(
            "Quality gate FAILED (retry %d/%d). Retrying query expansion...",
            retry_count + 1,
            settings.MAX_RETRIES,
        )
        return "expand_query"

    logger.warning(
        "Quality gate FAILED and max retries (%d) reached. Proceeding to format with warning.",
        settings.MAX_RETRIES,
    )
    return "format_response"


async def wrapped_expand_query(state: AgentState) -> dict[str, Any]:
    """Expand query and increment retry count if this is a retry."""
    res = await expand_query(state)
    # If we already had retrieved_docs, this is a retry loop
    if "retrieved_docs" in state and len(state.get("retrieved_docs", [])) > 0:
        res["retry_count"] = state.get("retry_count", 0) + 1
    return res


def build_graph() -> StateGraph:
    """Build and wire the LangGraph StateGraph for Muḥāwir."""
    workflow = StateGraph(AgentState)

    # Register nodes
    workflow.add_node("detect_language", trace_node("detect_language", detect_language))
    workflow.add_node("expand_query", trace_node("expand_query", wrapped_expand_query))
    workflow.add_node("retrieve", trace_node("retrieve", retrieve))
    workflow.add_node("rerank", trace_node("rerank", rerank))
    workflow.add_node("generate", trace_node("generate", generate))
    workflow.add_node("quality_gate", trace_node("quality_gate", quality_gate))
    workflow.add_node("format_response", trace_node("format_response", format_response))

    # Sequential edges
    workflow.set_entry_point("detect_language")
    workflow.add_edge("detect_language", "expand_query")
    workflow.add_edge("expand_query", "retrieve")
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "generate")
    workflow.add_edge("generate", "quality_gate")

    # Conditional branching
    workflow.add_conditional_edges(
        "quality_gate",
        check_quality_gate,
        {
            "format_response": "format_response",
            "expand_query": "expand_query",
        },
    )

    workflow.add_edge("format_response", END)

    return workflow


# Compiled stateless workflow
def compile_graph():
    """Compile graph with checkpointer=None (stateless per request)."""
    graph = build_graph()
    return graph.compile(checkpointer=None)
