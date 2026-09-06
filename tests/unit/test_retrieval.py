import pytest
from src.retrieval.chunking import chunk_documents, normalize_arabic
from src.retrieval.hybrid import BM25Index, reciprocal_rank_fusion


def test_bm25_search_returns_results(sample_documents):
    chunks = chunk_documents(sample_documents, chunk_size=200, overlap=20)
    index = BM25Index(corpus=chunks)

    # Search in Arabic
    results_ar = index.search("رؤية السعودية", top_k=5)
    assert len(results_ar) > 0
    assert any("doc_ar" in res[0] for res in results_ar)

    # Search in English
    results_en = index.search("Qatar Vision development", top_k=5)
    assert len(results_en) > 0
    assert any("doc_en" in res[0] for res in results_en)


def test_rrf_fusion_merges_correctly():
    list_a = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)]
    list_b = [("doc2", 0.95), ("doc1", 0.85), ("doc4", 0.6)]

    fused = reciprocal_rank_fusion([list_a, list_b], k=60)
    assert len(fused) == 4

    doc_ids = [item[0] for item in fused]
    # doc1 and doc2 appear in top 2 of both lists, should have highest combined RRF scores
    assert doc_ids[0] in ["doc1", "doc2"]
    assert doc_ids[1] in ["doc1", "doc2"]

    # Verify score calculation:
    # doc1: 1/(60+1) + 1/(60+2) = 1/61 + 1/62
    # doc2: 1/(60+2) + 1/(60+1) = 1/62 + 1/61 (tie)
    score_doc1 = next(score for did, score in fused if did == "doc1")
    expected_doc1 = (1.0 / 61) + (1.0 / 62)
    assert pytest.approx(score_doc1, rel=1e-5) == expected_doc1


def test_rrf_handles_empty_lists():
    fused = reciprocal_rank_fusion([], k=60)
    assert fused == []

    fused_partial = reciprocal_rank_fusion([[], [("doc1", 0.5)]], k=60)
    assert len(fused_partial) == 1
    assert fused_partial[0][0] == "doc1"


def test_arabic_normalization():
    raw_ar = "أَهْلاً وَسَهْلاً بِكُمْ فِي المَمْلَكَةِ العَرَبِيَّةِ السُّعُودِيَّةِ ـ2030"
    normalized = normalize_arabic(raw_ar)

    # Tashkeel removed
    assert "ً" not in normalized
    assert "ْ" not in normalized
    assert "َ" not in normalized
    # Alef variants normalized
    assert "أ" not in normalized
    assert "اهـ" not in normalized or "ـ" not in normalized
    # Tatweel removed
    assert "ـ" not in normalized


def test_chunking_respects_boundaries(sample_documents):
    chunks = chunk_documents(sample_documents, chunk_size=80, overlap=10)
    assert len(chunks) >= 2

    for chunk in chunks:
        assert "chunk_id" in chunk
        assert "doc_id" in chunk
        assert "text" in chunk
        assert len(chunk["text"]) > 0
        assert chunk["chunk_index"] >= 0
