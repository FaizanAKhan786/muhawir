from src.retrieval.chunking import chunk_documents, normalize_arabic


def test_alef_normalization():
    # Test all variations of alef (أ, إ, آ, ٱ) are unified to standard ا
    raw = "أحمد إبراهيم آمن ٱستثمار"
    normalized = normalize_arabic(raw)
    assert "أ" not in normalized
    assert "إ" not in normalized
    assert "آ" not in normalized
    assert "ٱ" not in normalized
    assert "احمد" in normalized
    assert "ابراهيم" in normalized
    assert "امن" in normalized
    assert "استثمار" in normalized


def test_tashkeel_removal():
    # Test all Arabic diacritical marks are stripped
    raw = "الرُّؤْيَةُ الاقْتِصَادِيَّةُ وَالتَّنْمِيَةُ الْمُسْتَدَامَةُ"
    normalized = normalize_arabic(raw)
    for char in ["ّ", "ُ", "ْ", "َ", "ِ", "ً", "ٍ", "ٌ"]:
        assert char not in normalized
    assert "الرؤية الاقتصادية والتنمية المستدامة" in normalized


def test_tatweel_removal():
    # Test kashida (ـ) removal
    raw = "مــــحـــــاور للــــذكـــــاء الاصــــطـنـاعي"
    normalized = normalize_arabic(raw)
    assert "ـ" not in normalized
    assert normalized == "محاور للذكاء الاصطناعي"


def test_mixed_ar_en_chunking():
    mixed_doc = [
        {
            "doc_id": "doc_mixed_test",
            "title": "Bilingual AI System محاور",
            "text": "يقوم نظام Muḥāwir بتحليل النصوص العربية والإنجليزية. The system uses LangGraph for multi-agent workflows. كما يعتمد على BM25 and Qdrant for hybrid search. This ensures high faithfulness.",
            "language": "mixed",
            "metadata": {"source": "mixed_guide.md"},
        }
    ]
    chunks = chunk_documents(mixed_doc, chunk_size=100, overlap=20)
    assert len(chunks) >= 1
    for c in chunks:
        assert c["language"] == "mixed"
        assert len(c["text"]) > 0
