"""Muḥāwir (محاور) - Production-Grade Bilingual Agentic Knowledge Assistant."""

__version__ = "1.2.0"

SYSTEM_CAPABILITIES: list[str] = [
    "bilingual_arabic_english_rag",
    "arabic_morphological_analysis_and_normalization",
    "hybrid_bm25_dense_retrieval",
    "cross_encoder_reranking",
    "multi_hop_react_reasoning",
    "self_reflection_query_rewriting",
    "stateful_conversation_memory",
    "coreference_resolution",
    "enterprise_guardrails_safety",
    "hallucination_factual_consistency_detection",
    "streaming_sse_support",
]


def get_version_info() -> dict:
    """Return application version, capabilities, and supported languages."""
    return {
        "name": "Muḥāwir (محاور)",
        "version": __version__,
        "description": "Production-grade Bilingual (Arabic/English) Agentic Knowledge Assistant",
        "capabilities": list(SYSTEM_CAPABILITIES),
        "supported_languages": ["ar", "en", "mixed"],
    }
