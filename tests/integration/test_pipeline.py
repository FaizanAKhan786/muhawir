import json
from unittest.mock import AsyncMock, MagicMock
import pytest

from src.agents.graph import compile_graph
from src.agents.nodes import set_openai_client, set_reranker
from src.agents.state import AgentState
from src.retrieval.cache import SemanticCache
from src.retrieval.chunking import chunk_documents
from src.retrieval.hybrid import BM25Index, set_retrieval_components
from src.retrieval.reranker import NeuralReranker


@pytest.fixture(autouse=True)
def setup_pipeline_environment(sample_documents):
    # Setup BM25 and mock LLM client
    chunks = chunk_documents(sample_documents)
    bm25 = BM25Index(corpus=chunks)
    doc_store = {c["chunk_id"]: c for c in chunks}
    set_retrieval_components(bm25=bm25, document_store=doc_store)

    # Mock CrossEncoder
    mock_reranker = MagicMock(spec=NeuralReranker)
    mock_reranker.rerank.side_effect = lambda query, documents, top_k=5: documents[:top_k]
    set_reranker(mock_reranker)

    # Mock AsyncOpenAI client
    mock_client = AsyncMock()

    async def mock_completion_create(*args, **kwargs):
        messages = kwargs.get("messages", [])
        prompt_text = " ".join([m.get("content", "") for m in messages])

        resp = MagicMock()
        choice = MagicMock()

        if "complexity classifier" in prompt_text:
            # Simple queries — route to single-hop path
            choice.message.content = json.dumps({
                "is_complex": False,
                "complexity_score": 0.25,
                "reasoning": "Simple factoid query",
                "estimated_hops": 1,
            })
        elif "search query variations" in prompt_text:
            choice.message.content = '["variation 1", "variation 2"]'
        elif "Is this answer fully supported?" in prompt_text:
            choice.message.content = '{"faithful": true, "language_ok": true, "score": 0.95}'
        elif "Answer using ONLY the provided context" in prompt_text:
            user_msg = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
            question = user_msg.split("Question:\n")[-1] if "Question:\n" in user_msg else user_msg
            if any("\u0600" <= c <= "\u06FF" for c in question):
                choice.message.content = json.dumps({
                    "answer": "تهدف رؤية السعودية 2030 إلى تنويع مصادر الدخل [Source 1].",
                    "language": "ar",
                    "citations": [{"doc_id": "doc_ar_1", "chunk_id": "doc_ar_1_c0", "text_snippet": "رؤية السعودية", "source": "saudi.txt"}],
                    "confidence": 0.95,
                })
            else:
                choice.message.content = json.dumps({
                    "answer": "Qatar National Vision 2030 is built upon four pillars [Source 1].",
                    "language": "en",
                    "citations": [{"doc_id": "doc_en_1", "chunk_id": "doc_en_1_c0", "text_snippet": "four pillars", "source": "qatar.txt"}],
                    "confidence": 0.95,
                })
        else:
            choice.message.content = "{}"

        resp.choices = [choice]
        return resp

    mock_client.chat.completions.create = AsyncMock(side_effect=mock_completion_create)
    set_openai_client(mock_client)


@pytest.mark.asyncio
async def test_full_pipeline_arabic_query():
    agent = compile_graph()
    initial_state: AgentState = {
        "query": "ما هي محاور رؤية السعودية 2030 وأهدافها؟",
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
        "trace_id": "test-pipe-ar",
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
        "sanitized_query": "ما هي محاور رؤية السعودية 2030 وأهدافها؟",
        "current_step": "",
        "steps_completed": [],
    }

    result = await agent.ainvoke(initial_state)

    assert result["language"] == "ar"
    assert len(result["retrieved_docs"]) > 0
    assert "رؤية السعودية" in result["final_answer"]
    assert result["quality_score"]["faithfulness"] >= 0.8
    assert result["retry_count"] == 0


@pytest.mark.asyncio
async def test_full_pipeline_english_query():
    agent = compile_graph()
    initial_state: AgentState = {
        "query": "What are the core pillars of Qatar National Vision 2030?",
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
        "trace_id": "test-pipe-en",
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
        "sanitized_query": "What are the core pillars of Qatar National Vision 2030?",
        "current_step": "",
        "steps_completed": [],
    }

    result = await agent.ainvoke(initial_state)

    assert result["language"] == "en"
    assert len(result["retrieved_docs"]) > 0
    assert "Qatar National Vision" in result["final_answer"]
    assert result["quality_score"]["faithfulness"] >= 0.8


@pytest.mark.asyncio
async def test_cache_hit_returns_cached(mock_redis):
    cache = SemanticCache(redis_client=mock_redis, threshold=0.90)

    query = "What is Qatar National Vision?"
    response_payload = {
        "answer": "Cached answer about Qatar National Vision.",
        "language": "en",
        "citations": [],
    }

    # Store in cache
    await cache.set(query, response_payload, ttl=3600)

    # Exact query hit
    hit = await cache.get(query)
    assert hit is not None
    assert hit["answer"] == response_payload["answer"]
