import functools
import logging
from typing import Any, Callable, Literal
from langgraph.graph import END, StateGraph

from src.agents.multi_hop import (
    classify_complexity,
    decompose_query,
    final_synthesize,
    hop_synthesize,
    multi_hop_rerank,
    multi_hop_retrieve,
)
from src.agents.nodes import (
    blocked_response,
    detect_language,
    expand_query,
    format_response,
    generate,
    input_guardrail,
    output_guardrail,
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


# ---------------------------------------------------------------------------
# Multi-hop routing decisions
# ---------------------------------------------------------------------------


def check_input_guardrail(state: AgentState) -> Literal["detect_language", "blocked_response"]:
    """Route after input_guardrail: pass or block."""
    if state.get("input_blocked", False):
        logger.warning("Input guardrail blocked query (trace_id=%s)", state.get("trace_id", ""))
        return "blocked_response"
    return "detect_language"


def route_by_complexity(state: AgentState) -> Literal["expand_query", "decompose_query"]:
    """After complexity classification, route simple queries to single-hop
    (expand_query) and complex queries to multi-hop (decompose_query)."""
    if state.get("is_multi_hop", False):
        logger.info(
            "Multi-hop path selected (complexity=%.2f, max_hops=%d)",
            state.get("complexity_score", 0.0),
            state.get("max_hops", 1),
        )
        return "decompose_query"
    logger.info("Single-hop path selected (complexity=%.2f)", state.get("complexity_score", 0.0))
    return "expand_query"


def route_after_hop_synthesis(state: AgentState) -> Literal["decompose_query", "final_synthesize"]:
    """After hop synthesis, decide whether to do another hop or finalize.

    Checks the internal ``_needs_another_hop`` flag set by hop_synthesize.
    """
    # The hop_synthesize node stores this transient flag
    needs_more = bool(state.get("_needs_another_hop", False))
    current_hop = state.get("current_hop", 1)
    max_hops = state.get("max_hops", 3)

    if needs_more and current_hop < max_hops:
        logger.info("Another hop needed (hop %d/%d)", current_hop + 1, max_hops)
        return "decompose_query"

    logger.info("All hops complete (%d hops). Proceeding to final synthesis.", current_hop)
    return "final_synthesize"


async def wrapped_expand_query(state: AgentState) -> dict[str, Any]:
    """Expand query and increment retry count if this is a retry."""
    res = await expand_query(state)
    # If we already had retrieved_docs, this is a retry loop
    if "retrieved_docs" in state and len(state.get("retrieved_docs", [])) > 0:
        res["retry_count"] = state.get("retry_count", 0) + 1
    return res


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_graph() -> StateGraph:
    """Build and wire the LangGraph StateGraph for Muḥāwir.

    Full pipeline topology:
    input_guardrail
        ├─ BLOCKED → blocked_response → END
        └─ PASS → detect_language → classify_complexity
                        ├─ simple → expand_query → retrieve → rerank
                        │           → output_guardrail → generate → quality_gate
                        │                └─ PASS → format_response → END
                        │                └─ FAIL/retry → expand_query (loop)
                        └─ complex → decompose_query → multi_hop_retrieve
                                    → multi_hop_rerank → hop_synthesize
                                        ├─ needs_hop → decompose_query (loop)
                                        └─ done → final_synthesize → output_guardrail_mh
                                                    → quality_gate_mh → format_response_mh → END
    """
    workflow = StateGraph(AgentState)

    # ---- Guardrail entry ----
    workflow.add_node("input_guardrail", trace_node("input_guardrail", input_guardrail))
    workflow.add_node("blocked_response", trace_node("blocked_response", blocked_response))

    # ---- Shared nodes ----
    workflow.add_node("detect_language", trace_node("detect_language", detect_language))
    workflow.add_node("classify_complexity", trace_node("classify_complexity", classify_complexity))

    # ---- Single-hop path (original) ----
    workflow.add_node("expand_query", trace_node("expand_query", wrapped_expand_query))
    workflow.add_node("retrieve", trace_node("retrieve", retrieve))
    workflow.add_node("rerank", trace_node("rerank", rerank))
    workflow.add_node("generate", trace_node("generate", generate))
    workflow.add_node("quality_gate", trace_node("quality_gate", quality_gate))
    workflow.add_node("format_response", trace_node("format_response", format_response))

    # ---- Multi-hop path ----
    workflow.add_node("decompose_query", trace_node("decompose_query", decompose_query))
    workflow.add_node("multi_hop_retrieve", trace_node("multi_hop_retrieve", multi_hop_retrieve))
    workflow.add_node("multi_hop_rerank", trace_node("multi_hop_rerank", multi_hop_rerank))
    workflow.add_node("hop_synthesize", trace_node("hop_synthesize", hop_synthesize))
    workflow.add_node("final_synthesize", trace_node("final_synthesize", final_synthesize))
    workflow.add_node("quality_gate_mh", trace_node("quality_gate_mh", quality_gate))
    workflow.add_node("format_response_mh", trace_node("format_response_mh", format_response))

    # ---- Output guardrail nodes (one per path) ----
    workflow.add_node("output_guardrail", trace_node("output_guardrail", output_guardrail))
    workflow.add_node("output_guardrail_mh", trace_node("output_guardrail_mh", output_guardrail))

    # ---- Wiring ----
    # Entry: input guardrail first
    workflow.set_entry_point("input_guardrail")
    workflow.add_conditional_edges(
        "input_guardrail",
        check_input_guardrail,
        {
            "detect_language": "detect_language",
            "blocked_response": "blocked_response",
        },
    )
    workflow.add_edge("blocked_response", END)

    # Language detection → complexity classification
    workflow.add_edge("detect_language", "classify_complexity")

    # Complexity routing: simple → expand_query, complex → decompose_query
    workflow.add_conditional_edges(
        "classify_complexity",
        route_by_complexity,
        {
            "expand_query": "expand_query",
            "decompose_query": "decompose_query",
        },
    )

    # Single-hop sequential edges
    workflow.add_edge("expand_query", "retrieve")
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "generate")
    workflow.add_edge("generate", "output_guardrail")
    workflow.add_edge("output_guardrail", "quality_gate")
    workflow.add_conditional_edges(
        "quality_gate",
        check_quality_gate,
        {
            "format_response": "format_response",
            "expand_query": "expand_query",
        },
    )
    workflow.add_edge("format_response", END)

    # Multi-hop sequential edges
    workflow.add_edge("decompose_query", "multi_hop_retrieve")
    workflow.add_edge("multi_hop_retrieve", "multi_hop_rerank")
    workflow.add_edge("multi_hop_rerank", "hop_synthesize")

    # Hop synthesis routing: another hop → decompose_query, done → final_synthesize
    workflow.add_conditional_edges(
        "hop_synthesize",
        route_after_hop_synthesis,
        {
            "decompose_query": "decompose_query",
            "final_synthesize": "final_synthesize",
        },
    )

    # Final synthesis → output guardrail → quality gate → format
    workflow.add_edge("final_synthesize", "output_guardrail_mh")
    workflow.add_edge("output_guardrail_mh", "quality_gate_mh")
    workflow.add_conditional_edges(
        "quality_gate_mh",
        check_quality_gate,
        {
            "format_response": "format_response_mh",
            "expand_query": "decompose_query",  # Re-decompose on quality failure
        },
    )
    workflow.add_edge("format_response_mh", END)

    return workflow


# Compiled stateless workflow
def compile_graph():
    """Compile graph with checkpointer=None (stateless per request)."""
    graph = build_graph()
    return graph.compile(checkpointer=None)
