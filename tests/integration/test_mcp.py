from datetime import datetime
import pytest
from src.mcp_servers.knowledge_base import get_document_content, search_documents
from src.mcp_servers.utilities import calculate, convert_hijri_gregorian, get_current_datetime
from src.retrieval.chunking import chunk_documents
from src.retrieval.hybrid import BM25Index, set_retrieval_components


@pytest.fixture(autouse=True)
def setup_mcp_store(sample_documents):
    chunks = chunk_documents(sample_documents)
    bm25 = BM25Index(corpus=chunks)
    doc_store = {c["chunk_id"]: c for c in chunks}
    for doc in sample_documents:
        doc_store[doc["doc_id"]] = doc
    set_retrieval_components(bm25=bm25, document_store=doc_store)


@pytest.mark.asyncio
async def test_kb_mcp_search():
    # Test valid query search
    res = await search_documents("Saudi Vision", language="en", top_k=3)
    assert isinstance(res, list)
    assert len(res) > 0
    assert "doc_id" in res[0]
    assert "score" in res[0]

    # Test empty query error handling
    err_res = await search_documents("")
    assert "error" in err_res[0]


@pytest.mark.asyncio
async def test_kb_mcp_get_document():
    # Existing document
    doc = await get_document_content("doc_ar_1")
    assert doc.get("doc_id") == "doc_ar_1"
    assert "رؤية" in doc.get("full_text", "")

    # Non-existent document
    missing = await get_document_content("non_existent_doc_id")
    assert "error" in missing


def test_utility_mcp_datetime():
    dt_str = get_current_datetime("Asia/Qatar")
    assert dt_str is not None
    # Validate ISO parsing
    parsed = datetime.fromisoformat(dt_str)
    assert parsed.tzinfo is not None


def test_utility_mcp_math_and_calendar():
    # Test safe calculation
    res = calculate("(150 * 4) / 2 + 10")
    assert float(res) == 310.0

    # Test Hijri-Gregorian conversions
    # 1445-09-01 is Ramadan 1445 AH (approx March 2024)
    greg_date = convert_hijri_gregorian("1445-09-01", direction="hijri_to_gregorian")
    assert greg_date.startswith("2024-")

    # Reverse
    hijri_back = convert_hijri_gregorian(greg_date, direction="gregorian_to_hijri")
    assert hijri_back.startswith("1445-")
