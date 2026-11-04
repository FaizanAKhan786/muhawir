import pytest
from src.agents.graph import check_quality_gate, route_by_complexity
from src.agents.nodes import detect_language
from src.agents.state import AgentState, QualityScore


_MH_DEFAULTS = {
    "is_multi_hop": False,
    "sub_queries": [],
    "reasoning_chain": [],
    "hop_contexts": [],
    "current_hop": 0,
    "max_hops": 3,
    "synthesis_answer": "",
    "complexity_score": 0.0,
    "guardrail_violations": [],
    "input_blocked": False,
    "output_sanitized": False,
    "sanitized_query": "",
    "current_step": "",
    "steps_completed": [],
}


def _state(**overrides) -> AgentState:
    """Build an AgentState with all required fields."""
    base: AgentState = {
        "query": "",
        "language": "en",
        "language_confidence": 0.0,
        "expanded_query": "",
        "retrieved_docs": [],
        "reranked_docs": [],
        "draft_answer": "",
        "quality_score": {"faithfulness": 1.0, "language_consistent": True, "grounded": True},
        "final_answer": "",
        "citations": [],
        "retry_count": 0,
        "error": None,
        "trace_id": "test-trace",
        **_MH_DEFAULTS,
    }
    base.update(overrides)
    return base


@pytest.mark.asyncio
async def test_detect_language_arabic():
    state = _state(query="ما هي ركائز رؤية قطر الوطنية وما أهدافها؟")
    result = await detect_language(state)
    assert result["language"] == "ar"
    assert result["language_confidence"] >= 0.7


@pytest.mark.asyncio
async def test_detect_language_english():
    state = _state(query="Explain how hybrid retrieval works with Reciprocal Rank Fusion.")
    result = await detect_language(state)
    assert result["language"] == "en"
    assert result["language_confidence"] >= 0.7


@pytest.mark.asyncio
async def test_detect_language_mixed():
    state = _state(query="كيف يمكن تحسين الـ hybrid retrieval باستخدام BM25 و Qdrant؟")
    result = await detect_language(state)
    assert result["language"] in ["mixed", "ar"]


def test_quality_gate_passes_on_high_score():
    state = _state(
        query="Sample query",
        language_confidence=0.9,
        draft_answer="Valid supported answer",
        quality_score={"faithfulness": 0.95, "language_consistent": True, "grounded": True},
    )
    decision = check_quality_gate(state)
    assert decision == "format_response"


def test_quality_gate_fails_on_low_score():
    state = _state(
        query="Sample query",
        language_confidence=0.9,
        draft_answer="Hallucinated answer",
        quality_score={"faithfulness": 0.50, "language_consistent": True, "grounded": False},
    )
    decision = check_quality_gate(state)
    assert decision == "expand_query"


def test_retry_on_quality_failure():
    # If retry_count reaches max retries (2), it should format_response despite failure
    state = _state(
        query="Sample query",
        language_confidence=0.9,
        draft_answer="Uncertain answer",
        quality_score={"faithfulness": 0.65, "language_consistent": True, "grounded": False},
        retry_count=2,  # Max retries reached
    )
    decision = check_quality_gate(state)
    assert decision == "format_response"


# ---------------------------------------------------------------------------
# Multi-hop routing tests
# ---------------------------------------------------------------------------


def test_route_by_complexity_selects_multi_hop():
    state = _state(is_multi_hop=True, complexity_score=0.85, max_hops=3)
    assert route_by_complexity(state) == "decompose_query"


def test_route_by_complexity_selects_single_hop():
    state = _state(is_multi_hop=False, complexity_score=0.3)
    assert route_by_complexity(state) == "expand_query"
