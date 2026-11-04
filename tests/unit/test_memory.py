"""Unit tests for conversation memory, long-term semantic profile store, and coreference resolution."""

import json
from unittest.mock import AsyncMock, MagicMock
import pytest

from src.agents.nodes import contextualize_query, set_openai_client
from src.agents.state import AgentState
from src.memory.coreference import is_contextual_query, resolve_coreference
from src.memory.long_term import LongTermMemoryStore, UserMemoryFact
from src.memory.short_term import (
    ConversationMessage,
    RedisConversationMemory,
    get_conversation_memory,
    set_conversation_memory,
)


# ---------------------------------------------------------------------------
# Short-Term Conversation Memory Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_short_term_memory_add_and_retrieve():
    """Verify adding messages and retrieving sliding window history."""
    mem = RedisConversationMemory(redis_client=None)  # Use in-memory store
    session_id = "test_sess_001"

    await mem.add_turn(
        session_id=session_id,
        user_query="What is Qatar National Vision 2030?",
        assistant_response="It is a development plan based on four pillars.",
    )

    history = await mem.get_history(session_id, max_messages=10)
    assert len(history) == 2
    assert history[0].role == "user"
    assert history[0].content == "What is Qatar National Vision 2030?"
    assert history[1].role == "assistant"
    assert "four pillars" in history[1].content


@pytest.mark.asyncio
async def test_short_term_memory_sliding_window():
    """Ensure sliding window returns only the most recent N messages."""
    mem = RedisConversationMemory(redis_client=None)
    session_id = "test_sess_sliding"

    # Add 4 turns (8 messages)
    for i in range(1, 5):
        await mem.add_turn(session_id, f"Question {i}", f"Answer {i}")

    # Request only last 4 messages (2 turns)
    history = await mem.get_history(session_id, max_messages=4)
    assert len(history) == 4
    assert history[0].content == "Question 3"
    assert history[-1].content == "Answer 4"


@pytest.mark.asyncio
async def test_short_term_memory_format_for_prompt():
    """Test formatting history into user/assistant transcript."""
    mem = RedisConversationMemory(redis_client=None)
    session_id = "test_format"

    await mem.add_turn(session_id, "Hello", "Greetings! How can I assist you?")
    prompt_str = await mem.format_history_for_prompt(session_id)

    assert "User: Hello" in prompt_str
    assert "Assistant: Greetings!" in prompt_str


@pytest.mark.asyncio
async def test_short_term_memory_clear_session():
    """Verify clearing session removes history."""
    mem = RedisConversationMemory(redis_client=None)
    session_id = "test_clear"

    await mem.add_turn(session_id, "Q", "A")
    assert len(await mem.get_history(session_id)) == 2

    cleared = await mem.clear_session(session_id)
    assert cleared is True
    assert len(await mem.get_history(session_id)) == 0


@pytest.mark.asyncio
async def test_short_term_memory_with_mock_redis(mock_redis):
    """Test RedisConversationMemory with a mocked Redis client."""
    mem = RedisConversationMemory(redis_client=mock_redis)
    session_id = "redis_sess"

    await mem.add_turn(session_id, "User Q", "Agent A")
    # Verify fallback / redis operations do not crash
    history = await mem.get_history(session_id)
    assert isinstance(history, list)


# ---------------------------------------------------------------------------
# Long-Term Semantic Memory Store Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_long_term_memory_save_and_retrieve():
    """Verify saving facts and semantic retrieval for a user."""
    store = LongTermMemoryStore(qdrant_client=None)
    user_id = "user_123"

    await store.save_fact(
        user_id=user_id,
        content="User is interested in renewable energy and sustainability initiatives in the Gulf.",
        category="topic_interest",
    )
    await store.save_fact(
        user_id=user_id,
        content="Prefers responses in Arabic with English technical terms preserved.",
        category="preference",
    )

    memories = await store.retrieve_user_memories(user_id=user_id, query="sustainability energy", top_k=2)
    assert len(memories) >= 1
    assert any("renewable energy" in m.content for m in memories)


@pytest.mark.asyncio
async def test_long_term_memory_prompt_formatting():
    """Verify formatting user memories into a system prompt section."""
    store = LongTermMemoryStore(qdrant_client=None)
    user_id = "user_prompt"

    await store.save_fact(user_id, "Interested in Qatar financial markets", "topic_interest")
    formatted = await store.format_memories_for_prompt(user_id, query="finance")

    assert "User Profile & Preferences:" in formatted
    assert "Qatar financial markets" in formatted


@pytest.mark.asyncio
async def test_long_term_memory_extract_facts():
    """Test LLM extraction of enduring facts from a dialogue turn."""
    store = LongTermMemoryStore(qdrant_client=None)
    user_id = "user_extract"

    mock_client = AsyncMock()
    mock_resp = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps([
        {"content": "User is conducting research on Saudi mining regulations", "category": "topic_interest"}
    ])
    mock_resp.choices = [mock_choice]
    mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

    facts = await store.extract_and_save_facts(
        user_id=user_id,
        user_query="I am a researcher studying the new Saudi mining regulations for my PhD thesis.",
        assistant_response="The Saudi mining investment law was enacted to facilitate extraction...",
        client=mock_client,
    )

    assert len(facts) == 1
    assert facts[0].category == "topic_interest"
    assert "mining regulations" in facts[0].content


# ---------------------------------------------------------------------------
# Coreference Resolution Tests
# ---------------------------------------------------------------------------


def test_is_contextual_query_heuristics():
    """Test heuristic detection of anaphoric pronouns and ellipsis."""
    # Contextual English queries
    assert is_contextual_query("What about its second pillar?") is True
    assert is_contextual_query("What are their goals?") is True
    assert is_contextual_query("Why is that?") is True
    assert is_contextual_query("Tell me more.") is True

    # Contextual Arabic queries
    assert is_contextual_query("وما هي ركائزها؟") is True
    assert is_contextual_query("ماذا عن أهدافها؟") is True
    assert is_contextual_query("قارن بينهما") is True

    # Standalone queries should NOT trigger coreference rewrite
    assert is_contextual_query("What are the four pillars of Qatar National Vision 2030?") is False
    assert is_contextual_query("ما هي رؤية المملكة العربية السعودية 2030؟") is False


@pytest.mark.asyncio
async def test_resolve_coreference_english():
    """Test rewriting English pronoun question using dialogue history."""
    history = [
        ConversationMessage(role="user", content="What is Qatar National Vision 2030?"),
        ConversationMessage(role="assistant", content="It is a long-term development framework."),
    ]

    mock_client = AsyncMock()
    mock_resp = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps({
        "rewritten_query": "What is the second pillar of Qatar National Vision 2030?",
        "is_rewritten": True,
        "referenced_topic": "Qatar National Vision 2030",
    })
    mock_resp.choices = [mock_choice]
    mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

    rewritten = await resolve_coreference(
        query="What is its second pillar?",
        history=history,
        language="en",
        client=mock_client,
    )

    assert rewritten == "What is the second pillar of Qatar National Vision 2030?"


@pytest.mark.asyncio
async def test_resolve_coreference_arabic():
    """Test rewriting Arabic pronoun question using dialogue history."""
    history = [
        ConversationMessage(role="user", content="ما هي رؤية السعودية 2030؟"),
        ConversationMessage(role="assistant", content="خطة وطنية لتنويع الاقتصاد وتطوير القطاعات."),
    ]

    mock_client = AsyncMock()
    mock_resp = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps({
        "rewritten_query": "ما هي الركائز الاقتصادية لرؤية السعودية 2030؟",
        "is_rewritten": True,
        "referenced_topic": "رؤية السعودية 2030",
    })
    mock_resp.choices = [mock_choice]
    mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

    rewritten = await resolve_coreference(
        query="وما هي ركائزها الاقتصادية؟",
        history=history,
        language="ar",
        client=mock_client,
    )

    assert rewritten == "ما هي الركائز الاقتصادية لرؤية السعودية 2030؟"


@pytest.mark.asyncio
async def test_resolve_coreference_bypasses_standalone_query():
    """Standalone queries should not invoke the LLM for coreference resolution."""
    mock_client = AsyncMock()

    rewritten = await resolve_coreference(
        query="What are the four pillars of Qatar National Vision 2030?",
        history=[ConversationMessage(role="user", content="Hi")],
        language="en",
        client=mock_client,
    )

    # LLM should not have been called
    mock_client.chat.completions.create.assert_not_called()
    assert rewritten == "What are the four pillars of Qatar National Vision 2030?"


# ---------------------------------------------------------------------------
# Agent Graph contextualize_query Node Test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_contextualize_query_node():
    """Test contextualize_query node in the agent pipeline."""
    # Pre-populate memory
    mem = RedisConversationMemory(redis_client=None)
    session_id = "node_test_sess"
    await mem.add_turn(
        session_id=session_id,
        user_query="Tell me about Qatar Vision 2030",
        assistant_response="Qatar National Vision 2030 has four pillars.",
    )
    set_conversation_memory(mem)

    # Mock LLM for coreference rewrite
    mock_client = AsyncMock()
    mock_resp = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps({
        "rewritten_query": "What are the four pillars of Qatar Vision 2030?",
        "is_rewritten": True,
        "referenced_topic": "Qatar Vision 2030",
    })
    mock_resp.choices = [mock_choice]
    mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)
    set_openai_client(mock_client)

    initial_state: AgentState = {
        "query": "What are its four pillars?",
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
        "trace_id": "test-ctx",
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
        "sanitized_query": "What are its four pillars?",
        "current_step": "",
        "steps_completed": [],
        "session_id": session_id,
        "user_id": None,
        "chat_history": [],
        "raw_query": "What are its four pillars?",
        "rewritten_query": "What are its four pillars?",
        "user_memories": [],
    }

    result = await contextualize_query(initial_state)

    assert result["raw_query"] == "What are its four pillars?"
    assert result["query"] == "What are the four pillars of Qatar Vision 2030?"
    assert len(result["chat_history"]) == 2
    assert len(result["reasoning_chain"]) >= 1
    assert "contextualize_query" in result["steps_completed"]
