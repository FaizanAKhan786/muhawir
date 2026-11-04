"""Unit tests for multi-hop reasoning engine."""

import json
from unittest.mock import AsyncMock, MagicMock
import pytest

from src.agents.multi_hop import (
    classify_complexity,
    decompose_query,
    final_synthesize,
    hop_synthesize,
    multi_hop_rerank,
    multi_hop_retrieve,
)
from src.agents.nodes import set_openai_client, set_reranker
from src.agents.state import AgentState
from src.retrieval.chunking import chunk_documents
from src.retrieval.hybrid import BM25Index, set_retrieval_components
from src.retrieval.reranker import NeuralReranker


def _make_base_state(**overrides) -> AgentState:
    """Create a base AgentState with all required fields."""
    base: AgentState = {
        "query": "Compare the economic goals of Saudi Vision 2030 and Qatar National Vision 2030",
        "language": "en",
        "language_confidence": 0.95,
        "expanded_query": "",
        "retrieved_docs": [],
        "reranked_docs": [],
        "draft_answer": "",
        "quality_score": {"faithfulness": 1.0, "language_consistent": True, "grounded": True},
        "final_answer": "",
        "citations": [],
        "retry_count": 0,
        "error": None,
        "trace_id": "test-multihop",
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
        "sanitized_query": "Compare the economic goals of Saudi Vision 2030 and Qatar National Vision 2030",
        "current_step": "",
        "steps_completed": [],
        "session_id": None,
        "user_id": None,
        "chat_history": [],
        "raw_query": "Compare the economic goals of Saudi Vision 2030 and Qatar National Vision 2030",
        "rewritten_query": "Compare the economic goals of Saudi Vision 2030 and Qatar National Vision 2030",
        "user_memories": [],
    }
    base.update(overrides)
    return base


@pytest.fixture(autouse=True)
def setup_multihop_env(sample_documents):
    """Set up retrieval components and mock LLM for multi-hop tests."""
    chunks = chunk_documents(sample_documents)
    bm25 = BM25Index(corpus=chunks)
    doc_store = {c["chunk_id"]: c for c in chunks}
    set_retrieval_components(bm25=bm25, document_store=doc_store)

    mock_reranker = MagicMock(spec=NeuralReranker)
    mock_reranker.rerank.side_effect = lambda query, documents, top_k=5: documents[:top_k]
    set_reranker(mock_reranker)

    mock_client = AsyncMock()

    async def mock_completion_create(*args, **kwargs):
        messages = kwargs.get("messages", [])
        prompt_text = " ".join([m.get("content", "") for m in messages])

        resp = MagicMock()
        choice = MagicMock()

        if "complexity classifier" in prompt_text:
            choice.message.content = json.dumps({
                "is_complex": True,
                "complexity_score": 0.85,
                "reasoning": "Comparison of two national visions requires multi-entity reasoning",
                "estimated_hops": 2,
            })
        elif "Break this complex query" in prompt_text:
            choice.message.content = json.dumps([
                {
                    "sub_query": "What are the economic goals of Saudi Vision 2030?",
                    "purpose": "Understand Saudi economic objectives",
                },
                {
                    "sub_query": "What are the economic goals of Qatar National Vision 2030?",
                    "purpose": "Understand Qatar economic objectives",
                },
            ])
        elif "comprehensive final answer" in prompt_text.lower() or "reasoning chain across" in prompt_text.lower():
            choice.message.content = json.dumps({
                "answer": "Both Saudi Vision 2030 and Qatar National Vision 2030 focus on economic diversification [Source 1] [Source 2].",
                "language": "en",
                "citations": [
                    {"doc_id": "doc_ar_1", "chunk_id": "doc_ar_1_c0", "text_snippet": "Saudi economic goals"},
                    {"doc_id": "doc_en_1", "chunk_id": "doc_en_1_c0", "text_snippet": "Qatar economic goals"},
                ],
                "confidence": 0.88,
                "reasoning_summary": "Compared economic goals across two national visions",
            })
        elif "you are on hop" in prompt_text.lower() or "sub-queries and their retrieved context" in prompt_text.lower():
            choice.message.content = json.dumps({
                "sub_answers": [
                    {
                        "sub_query": "What are the economic goals of Saudi Vision 2030?",
                        "answer": "Diversifying income sources and reducing oil dependency.",
                        "confidence": 0.85,
                        "needs_more_info": False,
                    },
                    {
                        "sub_query": "What are the economic goals of Qatar National Vision 2030?",
                        "answer": "Sustainable economic development through four pillars.",
                        "confidence": 0.80,
                        "needs_more_info": False,
                    },
                ],
                "partial_synthesis": "Both visions aim for economic diversification.",
                "needs_another_hop": False,
                "next_questions": [],
            })
        elif "search query variations" in prompt_text:
            choice.message.content = '["variation 1", "variation 2"]'
        elif "Is this answer fully supported?" in prompt_text:
            choice.message.content = '{"faithful": true, "language_ok": true, "score": 0.92}'
        else:
            choice.message.content = "{}"

        resp.choices = [choice]
        return resp

    mock_client.chat.completions.create = AsyncMock(side_effect=mock_completion_create)
    set_openai_client(mock_client)


# ---------------------------------------------------------------------------
# classify_complexity tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_classify_complexity_identifies_complex_query():
    """Complex comparison queries should be flagged for multi-hop."""
    state = _make_base_state(
        query="Compare the economic goals of Saudi Vision 2030 and Qatar National Vision 2030"
    )
    result = await classify_complexity(state)

    assert result["is_multi_hop"] is True
    assert result["complexity_score"] >= 0.6
    assert result["max_hops"] >= 2
    assert len(result["reasoning_chain"]) == 1
    assert result["reasoning_chain"][0]["action"] == "decompose"


@pytest.mark.asyncio
async def test_classify_complexity_simple_query():
    """Simple factoid queries should bypass multi-hop."""
    # Override the mock to return simple classification
    from src.agents.nodes import get_openai_client

    client = get_openai_client()
    original_create = client.chat.completions.create

    async def simple_mock(*args, **kwargs):
        messages = kwargs.get("messages", [])
        prompt_text = " ".join([m.get("content", "") for m in messages])
        resp = MagicMock()
        choice = MagicMock()

        if "complexity classifier" in prompt_text:
            choice.message.content = json.dumps({
                "is_complex": False,
                "complexity_score": 0.25,
                "reasoning": "Simple factoid lookup",
                "estimated_hops": 1,
            })
        else:
            return await original_create(*args, **kwargs)

        resp.choices = [choice]
        return resp

    client.chat.completions.create = AsyncMock(side_effect=simple_mock)

    state = _make_base_state(query="What is Qatar National Vision 2030?")
    result = await classify_complexity(state)

    assert result["is_multi_hop"] is False
    assert result["complexity_score"] < 0.6

    # Restore original mock
    client.chat.completions.create = original_create


# ---------------------------------------------------------------------------
# decompose_query tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_decompose_query_produces_sub_queries():
    """Decomposition should generate multiple targeted sub-queries."""
    state = _make_base_state(
        query="Compare the economic goals of Saudi Vision 2030 and Qatar National Vision 2030",
        is_multi_hop=True,
        current_hop=0,
    )
    result = await decompose_query(state)

    assert len(result["sub_queries"]) >= 2
    assert result["current_hop"] == 1
    for sq in result["sub_queries"]:
        assert sq["sub_query"] != ""
        assert sq["purpose"] != ""
        assert sq["answered"] is False
        assert sq["hop_number"] == 1


@pytest.mark.asyncio
async def test_decompose_query_uses_prior_context():
    """Decomposition on hop 2+ should reference previous findings."""
    state = _make_base_state(
        current_hop=1,
        hop_contexts=[{
            "hop_number": 1,
            "query": "test",
            "retrieved_docs": [],
            "reranked_docs": [],
            "partial_answer": "Saudi Vision focuses on diversification.",
            "confidence": 0.8,
        }],
    )
    result = await decompose_query(state)

    assert result["current_hop"] == 2
    assert len(result["sub_queries"]) >= 1


# ---------------------------------------------------------------------------
# multi_hop_retrieve tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multi_hop_retrieve_fetches_for_all_sub_queries():
    """Retrieval should run for each unanswered sub-query."""
    state = _make_base_state(
        sub_queries=[
            {
                "sub_query": "Saudi Vision 2030 economic goals",
                "purpose": "Get Saudi goals",
                "answered": False,
                "answer": "",
                "supporting_docs": [],
                "hop_number": 1,
            },
            {
                "sub_query": "Qatar National Vision 2030 economic goals",
                "purpose": "Get Qatar goals",
                "answered": False,
                "answer": "",
                "supporting_docs": [],
                "hop_number": 1,
            },
        ]
    )

    result = await multi_hop_retrieve(state)

    assert len(result["retrieved_docs"]) > 0
    assert len(result["sub_queries"]) == 2
    # Each sub-query should have supporting docs
    for sq in result["sub_queries"]:
        assert len(sq["supporting_docs"]) > 0
    assert len(result["reasoning_chain"]) == 1
    assert result["reasoning_chain"][0]["action"] == "retrieve"


@pytest.mark.asyncio
async def test_multi_hop_retrieve_skips_answered():
    """Already-answered sub-queries should not be re-retrieved."""
    state = _make_base_state(
        sub_queries=[
            {
                "sub_query": "Saudi goals",
                "purpose": "Get Saudi goals",
                "answered": True,
                "answer": "Already answered",
                "supporting_docs": [],
                "hop_number": 1,
            },
        ]
    )

    result = await multi_hop_retrieve(state)
    # The answered sub-query keeps its state
    assert result["sub_queries"][0]["answered"] is True


# ---------------------------------------------------------------------------
# multi_hop_rerank tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multi_hop_rerank_produces_global_ranking():
    """Reranking should produce a global doc ranking from all sub-query docs."""
    sample_docs = [
        {
            "doc_id": "d1", "chunk_id": "d1_c0", "text": "Saudi economic diversification",
            "source": "s1", "score": 0.9, "language": "en", "metadata": {},
        },
        {
            "doc_id": "d2", "chunk_id": "d2_c0", "text": "Qatar sustainable development",
            "source": "s2", "score": 0.85, "language": "en", "metadata": {},
        },
    ]

    state = _make_base_state(
        sub_queries=[
            {
                "sub_query": "Saudi economic goals",
                "purpose": "test",
                "answered": False,
                "answer": "",
                "supporting_docs": sample_docs[:1],
                "hop_number": 1,
            },
            {
                "sub_query": "Qatar economic goals",
                "purpose": "test",
                "answered": False,
                "answer": "",
                "supporting_docs": sample_docs[1:],
                "hop_number": 1,
            },
        ],
        retrieved_docs=sample_docs,
    )

    result = await multi_hop_rerank(state)
    assert len(result["reranked_docs"]) >= 1
    assert len(result["sub_queries"]) == 2


# ---------------------------------------------------------------------------
# hop_synthesize tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hop_synthesize_creates_hop_context():
    """Hop synthesis should create a HopContext record."""
    state = _make_base_state(
        current_hop=1,
        max_hops=2,
        sub_queries=[
            {
                "sub_query": "What are the economic goals of Saudi Vision 2030?",
                "purpose": "Understand Saudi economic objectives",
                "answered": False,
                "answer": "",
                "supporting_docs": [
                    {
                        "doc_id": "d1", "chunk_id": "d1_c0",
                        "text": "Saudi Vision 2030 aims to diversify income.",
                        "source": "s1", "score": 0.9, "language": "en", "metadata": {},
                    }
                ],
                "hop_number": 1,
            },
        ],
        hop_contexts=[],
    )

    result = await hop_synthesize(state)

    assert len(result["hop_contexts"]) == 1
    assert result["hop_contexts"][0]["hop_number"] == 1
    assert result["hop_contexts"][0]["partial_answer"] != ""


# ---------------------------------------------------------------------------
# final_synthesize tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_final_synthesize_produces_answer_with_citations():
    """Final synthesis should produce a draft answer with citations."""
    state = _make_base_state(
        hop_contexts=[
            {
                "hop_number": 1,
                "query": "Compare economic goals",
                "retrieved_docs": [],
                "reranked_docs": [
                    {
                        "doc_id": "d1", "chunk_id": "d1_c0",
                        "text": "Saudi Vision 2030 aims to diversify income.",
                        "source": "s1", "score": 0.9, "language": "en", "metadata": {},
                    },
                    {
                        "doc_id": "d2", "chunk_id": "d2_c0",
                        "text": "Qatar National Vision 2030 sustainable development.",
                        "source": "s2", "score": 0.85, "language": "en", "metadata": {},
                    },
                ],
                "partial_answer": "Both visions aim for diversification.",
                "confidence": 0.85,
            },
        ],
        reranked_docs=[],
    )

    result = await final_synthesize(state)

    assert result["draft_answer"] != ""
    assert len(result["citations"]) >= 1
    assert len(result["reasoning_chain"]) >= 1
    assert result["reasoning_chain"][-1]["action"] == "synthesize"
