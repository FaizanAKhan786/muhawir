import asyncio
import inspect
from typing import Any
from unittest.mock import AsyncMock, MagicMock
import pytest


def pytest_pyfunc_call(pyfuncitem):
    """Run async test functions using asyncio.run if pytest-asyncio plugin is not active."""
    if inspect.iscoroutinefunction(pyfuncitem.obj):
        testargs = {
            arg: pyfuncitem.funcargs[arg]
            for arg in pyfuncitem._fixtureinfo.argnames
            if arg in pyfuncitem.funcargs
        }
        asyncio.run(pyfuncitem.obj(**testargs))
        return True
    return None


@pytest.fixture
def mock_qdrant_client():
    """Mock QdrantClient fixture returning synthetic search hits."""
    client = MagicMock()
    hit1 = MagicMock()
    hit1.id = "doc_1_c0"
    hit1.score = 0.92
    hit1.payload = {
        "chunk_id": "doc_1_c0",
        "doc_id": "doc_1",
        "text": "تعتمد رؤية السعودية 2030 على تنويع مصادر الاقتصاد.",
        "language": "ar",
        "metadata": {"source": "saudi_vision.txt"},
    }

    hit2 = MagicMock()
    hit2.id = "doc_2_c0"
    hit2.score = 0.85
    hit2.payload = {
        "chunk_id": "doc_2_c0",
        "doc_id": "doc_2",
        "text": "Qatar National Vision 2030 aims to foster sustainable development.",
        "language": "en",
        "metadata": {"source": "qatar_vision.txt"},
    }

    client.search.return_value = [hit1, hit2]
    client.get_collections.return_value.collections = [MagicMock(name="muhawir_docs")]
    return client


@pytest.fixture
def mock_redis():
    """Mock Redis client with in-memory store."""
    client = MagicMock()
    store: dict[str, Any] = {}
    sets: dict[str, set] = {}

    async def async_get(key):
        return store.get(key)

    async def async_set(key, val, ex=None):
        store[key] = val
        return True

    async def async_setex(key, ttl, val):
        store[key] = val
        return True

    async def async_sadd(set_key, member):
        if set_key not in sets:
            sets[set_key] = set()
        sets[set_key].add(member)
        return 1

    async def async_smembers(set_key):
        return list(sets.get(set_key, set()))

    async def async_ping():
        return True

    async def async_delete(key):
        store.pop(key, None)
        sets.pop(key, None)
        return 1

    client.get = AsyncMock(side_effect=async_get)
    client.set = AsyncMock(side_effect=async_set)
    client.setex = AsyncMock(side_effect=async_setex)
    client.sadd = AsyncMock(side_effect=async_sadd)
    client.smembers = AsyncMock(side_effect=async_smembers)
    client.delete = AsyncMock(side_effect=async_delete)
    client.ping = AsyncMock(side_effect=async_ping)
    return client


@pytest.fixture
def sample_documents() -> list[dict[str, Any]]:
    """Sample documents for testing chunking, normalization, and retrieval."""
    return [
        {
            "doc_id": "doc_ar_1",
            "title": "رؤية المملكة 2030",
            "text": "رؤية السعودية 2030 هي خطة وطنية كبرى. تهدف إلى تنويع مصادر الدخل، ودعم الاقتصاد المعرفي، وتطوير السياحة.",
            "language": "ar",
            "metadata": {"source": "saudi.txt"},
        },
        {
            "doc_id": "doc_en_1",
            "title": "Qatar National Vision",
            "text": "Qatar National Vision 2030 is built upon four interconnected pillars. Human, social, economic, and environmental development are central goals.",
            "language": "en",
            "metadata": {"source": "qatar.txt"},
        },
    ]


@pytest.fixture
def sample_golden_set() -> list[dict[str, Any]]:
    """Small golden set fixture for unit tests."""
    return [
        {
            "id": "test_ar_1",
            "question": "ما هي رؤية السعودية 2030؟",
            "ground_truth": "خطة وطنية لتنويع الاقتصاد وتقليل الاعتماد على النفط.",
            "contexts": ["رؤية السعودية 2030 هي خطة وطنية كبرى تهدف إلى تنويع مصادر الدخل."],
            "language": "ar",
            "expected_doc_id": "doc_ar_1",
        },
        {
            "id": "test_en_1",
            "question": "What are the four pillars of Qatar National Vision 2030?",
            "ground_truth": "Human, social, economic, and environmental development.",
            "contexts": ["The pillars are human, social, economic, and environmental development."],
            "language": "en",
            "expected_doc_id": "doc_en_1",
        },
    ]


@pytest.fixture
def mock_vllm_response():
    """Mock vLLM OpenAI client response."""
    mock_client = AsyncMock()
    mock_completion = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = '{"answer": "Answer based on context [Source 1].", "language": "en", "citations": [{"doc_id": "doc_en_1", "chunk_id": "doc_en_1_c0", "text_snippet": "Pillars are human, social, economic", "source": "qatar.txt"}], "confidence": 0.95}'
    mock_completion.choices = [mock_choice]
    mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)
    return mock_client
