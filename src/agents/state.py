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


class SubQuery(TypedDict):
    sub_query: str
    purpose: str
    answered: bool
    answer: str
    supporting_docs: list[RetrievedDocument]
    hop_number: int


class ReasoningStep(TypedDict):
    step_number: int
    action: Literal["decompose", "retrieve", "synthesize", "verify"]
    input_text: str
    output_text: str
    confidence: float


class HopContext(TypedDict):
    hop_number: int
    query: str
    retrieved_docs: list[RetrievedDocument]
    reranked_docs: list[RetrievedDocument]
    partial_answer: str
    confidence: float


class GuardrailViolation(TypedDict):
    """Records a single guardrail policy violation."""

    stage: str          # "input" or "output"
    check: str          # name of the check that fired
    description: str    # human-readable explanation


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
    is_multi_hop: bool
    sub_queries: list[SubQuery]
    reasoning_chain: list[ReasoningStep]
    hop_contexts: list[HopContext]
    current_hop: int
    max_hops: int
    synthesis_answer: str
    complexity_score: float
    # Guardrail fields
    guardrail_violations: list[GuardrailViolation]
    input_blocked: bool          # True if input guardrails rejected the query
    output_sanitized: bool       # True if output was modified by output guards
    sanitized_query: str         # PII-redacted version of the original query
    # Streaming step visibility
    current_step: str            # Human-readable current pipeline stage name
    steps_completed: list[str]   # Ordered list of completed step names
    # Stateful memory & coreference fields
    session_id: str | None       # Active session identifier
    user_id: str | None          # End-user identifier for long-term memory
    chat_history: list[dict[str, Any]] # Recent conversation messages
    raw_query: str               # Original unedited user query
    rewritten_query: str         # Standalone query after coreference resolution
    user_memories: list[dict[str, Any]] # Enduring user facts retrieved from semantic store
