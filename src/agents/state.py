from typing import Any, Literal
from typing_extensions import TypedDict


class RetrievedDocument(TypedDict):
    doc_id: str
    chunk_id: str
    text: str
    source: str
    score: float
    language: str
    metadata: dict[str, Any]


class Citation(TypedDict):
    doc_id: str
    chunk_id: str
    text_snippet: str
    source: str


class QualityScore(TypedDict):
    faithfulness: float
    language_consistent: bool
    grounded: bool


class AgentState(TypedDict):
    query: str
    language: Literal["ar", "en", "mixed"]
    language_confidence: float
    expanded_query: str
    retrieved_docs: list[RetrievedDocument]
    reranked_docs: list[RetrievedDocument]
    draft_answer: str
    quality_score: QualityScore
    final_answer: str
    citations: list[Citation]
    retry_count: int
    error: str | None
    trace_id: str
