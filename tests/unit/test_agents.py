import pytest
from src.agents.graph import check_quality_gate
from src.agents.nodes import detect_language
from src.agents.state import AgentState, QualityScore


@pytest.mark.asyncio
async def test_detect_language_arabic():
    state: AgentState = {
        "query": "ما هي ركائز رؤية قطر الوطنية وما أهدافها؟",
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
    }
    result = await detect_language(state)
    assert result["language"] == "ar"
    assert result["language_confidence"] >= 0.7


@pytest.mark.asyncio
async def test_detect_language_english():
    state: AgentState = {
        "query": "Explain how hybrid retrieval works with Reciprocal Rank Fusion.",
        "language": "ar",
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
    }
    result = await detect_language(state)
    assert result["language"] == "en"
    assert result["language_confidence"] >= 0.7


@pytest.mark.asyncio
async def test_detect_language_mixed():
    state: AgentState = {
        "query": "كيف يمكن تحسين الـ hybrid retrieval باستخدام BM25 و Qdrant؟",
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
    }
    result = await detect_language(state)
    assert result["language"] in ["mixed", "ar"]


def test_quality_gate_passes_on_high_score():
    state: AgentState = {
        "query": "Sample query",
        "language": "en",
        "language_confidence": 0.9,
        "expanded_query": "",
        "retrieved_docs": [],
        "reranked_docs": [],
        "draft_answer": "Valid supported answer",
        "quality_score": {"faithfulness": 0.95, "language_consistent": True, "grounded": True},
        "final_answer": "",
        "citations": [],
        "retry_count": 0,
        "error": None,
        "trace_id": "test-trace",
    }
    decision = check_quality_gate(state)
    assert decision == "format_response"


def test_quality_gate_fails_on_low_score():
    state: AgentState = {
        "query": "Sample query",
        "language": "en",
        "language_confidence": 0.9,
        "expanded_query": "",
        "retrieved_docs": [],
        "reranked_docs": [],
        "draft_answer": "Hallucinated answer",
        "quality_score": {"faithfulness": 0.50, "language_consistent": True, "grounded": False},
        "final_answer": "",
        "citations": [],
        "retry_count": 0,
        "error": None,
        "trace_id": "test-trace",
    }
    decision = check_quality_gate(state)
    assert decision == "expand_query"


def test_retry_on_quality_failure():
    # If retry_count reaches max retries (2), it should format_response despite failure
    state: AgentState = {
        "query": "Sample query",
        "language": "en",
        "language_confidence": 0.9,
        "expanded_query": "",
        "retrieved_docs": [],
        "reranked_docs": [],
        "draft_answer": "Uncertain answer",
        "quality_score": {"faithfulness": 0.65, "language_consistent": True, "grounded": False},
        "final_answer": "",
        "citations": [],
        "retry_count": 2,  # Max retries reached
        "error": None,
        "trace_id": "test-trace",
    }
    decision = check_quality_gate(state)
    assert decision == "format_response"
