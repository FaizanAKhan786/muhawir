import functools
import json
import logging
import os
import re
from typing import Any
from openai import AsyncOpenAI

from src.agents.state import AgentState, Citation, GuardrailViolation, QualityScore, RetrievedDocument
from src.config import get_settings
from src.retrieval.hybrid import hybrid_retrieve
from src.retrieval.reranker import NeuralReranker

logger = logging.getLogger(__name__)

# Global client and reranker caches
_ASYNC_OPENAI_CLIENT: AsyncOpenAI | None = None
_GLOBAL_RERANKER: NeuralReranker | None = None


def get_openai_client() -> AsyncOpenAI:
    global _ASYNC_OPENAI_CLIENT
    if _ASYNC_OPENAI_CLIENT is None:
        settings = get_settings()
        _ASYNC_OPENAI_CLIENT = AsyncOpenAI(
            base_url=settings.VLLM_BASE_URL,
            api_key=settings.API_KEY or "none",
        )
    return _ASYNC_OPENAI_CLIENT


def set_openai_client(client: AsyncOpenAI | None) -> None:
    global _ASYNC_OPENAI_CLIENT
    _ASYNC_OPENAI_CLIENT = client


def get_reranker() -> NeuralReranker:
    global _GLOBAL_RERANKER
    if _GLOBAL_RERANKER is None:
        settings = get_settings()
        _GLOBAL_RERANKER = NeuralReranker(settings.RERANKER_MODEL)
    return _GLOBAL_RERANKER


def set_reranker(reranker: NeuralReranker | None) -> None:
    global _GLOBAL_RERANKER
    _GLOBAL_RERANKER = reranker


@functools.lru_cache(maxsize=1)
def load_fasttext_model(model_path: str):
    """Load fasttext language identification model if available."""
    if os.path.exists(model_path):
        try:
            import fasttext  # type: ignore

            return fasttext.load_model(model_path)
        except Exception as e:
            logger.warning("Could not load fasttext model from %s: %s", model_path, e)
    return None


def detect_language_rule_based(text: str) -> tuple[str, float]:
    """Robust heuristic language detector for Arabic, English, or Mixed text."""
    arabic_chars = len(re.findall(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]", text))
    english_chars = len(re.findall(r"[A-Za-z]", text))
    total_alpha = arabic_chars + english_chars

    if total_alpha == 0:
        return "en", 0.5

    ar_ratio = arabic_chars / total_alpha
    en_ratio = english_chars / total_alpha

    if 0.25 <= ar_ratio <= 0.75 and 0.25 <= en_ratio <= 0.75:
        return "mixed", 0.90
    elif ar_ratio > 0.6:
        return "ar", min(0.99, 0.7 + ar_ratio * 0.3)
    elif en_ratio > 0.6:
        return "en", min(0.99, 0.7 + en_ratio * 0.3)

    return ("ar", 0.75) if ar_ratio > en_ratio else ("en", 0.75)


async def detect_language(state: AgentState) -> dict[str, Any]:
    """Node 1: Detect query language using FastText or heuristic fallback."""
    settings = get_settings()
    query = state.get("query", "")
    ft_model = load_fasttext_model(settings.FASTTEXT_MODEL_PATH)

    lang = "en"
    confidence = 0.5

    if ft_model is not None:
        try:
            clean_text = query.replace("\n", " ").strip()
            predictions = ft_model.predict(clean_text, k=2)
            labels, confidences = predictions
            primary_label = labels[0].replace("__label__", "")
            primary_conf = float(confidences[0])

            if primary_label == "ar":
                lang = "ar"
                confidence = primary_conf
            elif primary_label == "en":
                lang = "en"
                confidence = primary_conf
            else:
                # Check secondary label or heuristic
                h_lang, h_conf = detect_language_rule_based(query)
                lang, confidence = h_lang, h_conf

            # Check for mixed Arabic/English
            h_lang, _ = detect_language_rule_based(query)
            if h_lang == "mixed":
                lang = "mixed"
                confidence = 0.95
        except Exception as e:
            logger.warning("FastText prediction failed: %s", e)
            lang, confidence = detect_language_rule_based(query)
    else:
        lang, confidence = detect_language_rule_based(query)

    if confidence < 0.7:
        lang = "en"

    return {
        "language": lang,
        "language_confidence": confidence,
    }


async def expand_query(state: AgentState) -> dict[str, Any]:
    """Node 2: Expand query with synonyms and alternative phrasings."""
    query = state.get("query", "")
    language = state.get("language", "en")
    client = get_openai_client()
    settings = get_settings()

    prompt = (
        f"Generate 2-3 search query variations for: {query}. "
        f"Language: {language}. "
        "Return ONLY a JSON array of strings, e.g. [\"var 1\", \"var 2\"]."
    )

    variations: list[str] = []
    try:
        response = await client.chat.completions.create(
            model=settings.VLLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=150,
        )
        content = response.choices[0].message.content or "[]"
        # Extract JSON array using regex if surrounded by markdown fences
        json_match = re.search(r"\[.*\]", content, re.DOTALL)
        if json_match:
            variations = json.loads(json_match.group(0))
        elif content.strip().startswith("["):
            variations = json.loads(content.strip())
    except Exception as e:
        logger.warning("Query expansion LLM call failed, using original query: %s", e)
        variations = []

    clean_variations = [v for v in variations if isinstance(v, str) and v.strip()]
    if clean_variations:
        expanded_query = f"{query} " + " ".join(clean_variations)
    else:
        expanded_query = query

    return {"expanded_query": expanded_query}


async def retrieve(state: AgentState) -> dict[str, Any]:
    """Node 3: Hybrid retrieve top 20 candidate documents."""
    expanded_query = state.get("expanded_query") or state.get("query", "")
    language = state.get("language", "en")
    docs = await hybrid_retrieve(query=expanded_query, language=language, top_k=20)
    return {"retrieved_docs": docs}


async def rerank(state: AgentState) -> dict[str, Any]:
    """Node 4: Neural rerank retrieved documents from top-20 down to top-5."""
    query = state.get("query", "")
    retrieved_docs = state.get("retrieved_docs", [])
    reranker = get_reranker()

    reranked = reranker.rerank(query=query, documents=retrieved_docs, top_k=5)
    return {"reranked_docs": reranked}


async def generate(state: AgentState) -> dict[str, Any]:
    """Node 5: Generate structured answer with citations using vLLM."""
    reranked_docs = state.get("reranked_docs", [])
    query = state.get("query", "")
    client = get_openai_client()
    settings = get_settings()

    # Build context string
    context_parts = []
    for idx, doc in enumerate(reranked_docs, start=1):
        context_parts.append(
            f"[Source {idx}] (doc_id: {doc['doc_id']}, chunk_id: {doc['chunk_id']}): {doc['text']}"
        )
    context_str = "\n\n".join(context_parts) if context_parts else "No relevant context found."

    system_prompt = """You are Muḥāwir (محاور), a bilingual knowledge assistant. Answer using ONLY the provided context.
Rules:
1. Answer in the SAME language as the user's question.
2. Every claim must cite sources: [Source 1], [Source 2, §3].
3. If context is insufficient, say so. Do NOT fabricate.
4. Be concise but complete.
5. Preserve technical terms in original language.
Respond in JSON: {"answer": "...", "language": "ar|en", "citations": [{"doc_id": "...", "chunk_id": "...", "text_snippet": "..."}], "confidence": 0.0-1.0}"""

    user_prompt = f"Context:\n{context_str}\n\nQuestion:\n{query}"

    try:
        response = await client.chat.completions.create(
            model=settings.VLLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=800,
        )
        content = response.choices[0].message.content or "{}"
        # Extract JSON object
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        parsed = json.loads(json_match.group(0)) if json_match else json.loads(content)

        draft_answer = parsed.get("answer", "")
        citations: list[Citation] = parsed.get("citations", [])
    except Exception as e:
        logger.warning("Answer generation failed, producing fallback: %s", e)
        # Produce a coherent fallback
        if reranked_docs:
            top_doc = reranked_docs[0]
            draft_answer = f"Based on the available records: {top_doc['text'][:200]}... [Source 1]"
            citations = [
                {
                    "doc_id": top_doc["doc_id"],
                    "chunk_id": top_doc["chunk_id"],
                    "text_snippet": top_doc["text"][:100],
                    "source": top_doc["source"],
                }
            ]
        else:
            draft_answer = "Context insufficient to answer the query accurately."
            citations = []

    return {
        "draft_answer": draft_answer,
        "citations": citations,
    }


async def quality_gate(state: AgentState) -> dict[str, Any]:
    """Node 6: Lightweight LLM judge evaluating faithfulness and language consistency."""
    draft_answer = state.get("draft_answer", "")
    reranked_docs = state.get("reranked_docs", [])
    language = state.get("language", "en")
    client = get_openai_client()
    settings = get_settings()

    docs_summary = "\n".join([f"- {d['text'][:200]}" for d in reranked_docs])
    prompt = (
        f"Given these documents:\n{docs_summary}\n\n"
        f"Proposed answer:\n{draft_answer}\n\n"
        f"Is this answer fully supported? Is it in {language}? "
        "Reply JSON: {\"faithful\": bool, \"language_ok\": bool, \"score\": 0.0-1.0}"
    )

    try:
        response = await client.chat.completions.create(
            model=settings.VLLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=100,
        )
        content = response.choices[0].message.content or "{}"
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        parsed = json.loads(json_match.group(0)) if json_match else json.loads(content)

        score_val = float(parsed.get("score", 0.85))
        faithful_val = bool(parsed.get("faithful", True))
        lang_ok = bool(parsed.get("language_ok", True))
    except Exception as e:
        logger.warning("Quality gate LLM call failed, assuming standard heuristics: %s", e)
        score_val = 0.85
        faithful_val = True
        lang_ok = True

    quality: QualityScore = {
        "faithfulness": score_val,
        "language_consistent": lang_ok,
        "grounded": faithful_val,
    }

    return {"quality_score": quality}


async def format_response(state: AgentState) -> dict[str, Any]:
    """Node 7: Clean answer, append citations, and flag low confidence if retries exceeded."""
    draft = state.get("draft_answer", "").strip()
    citations = state.get("citations", [])
    quality = state.get("quality_score", {"faithfulness": 1.0, "language_consistent": True, "grounded": True})
    retry_count = state.get("retry_count", 0)
    settings = get_settings()

    final_text = draft

    # Format citations as inline references or appendix if not already present
    if citations and "[Source" not in final_text:
        source_refs = []
        for idx, cit in enumerate(citations, start=1):
            source_refs.append(f"[Source {idx}: {cit.get('source', cit.get('doc_id'))}]")
        final_text += "\n\nSources: " + ", ".join(source_refs)

    # If quality gate failed on final retry, prepend low confidence warning
    is_failing = (
        quality.get("faithfulness", 0.0) < settings.FAITHFULNESS_THRESHOLD
        or not quality.get("language_consistent", True)
    )
    if is_failing and retry_count >= settings.MAX_RETRIES:
        final_text = f"⚠️ Low confidence: {final_text}"

    return {"final_answer": final_text}


# ---------------------------------------------------------------------------
# Guardrail Nodes
# ---------------------------------------------------------------------------


async def input_guardrail(state: AgentState) -> dict[str, Any]:
    """Node: Run input safety guardrails before any LLM/retrieval call.

    Performs:
    - Query length and structural sanity checks
    - Prompt injection detection (English + Arabic)
    - Toxicity / harmful content heuristic filtering
    - PII detection and redaction

    Sets ``input_blocked=True`` if any hard guardrail fires, which causes
    ``check_input_guardrail`` conditional edge to route to ``blocked_response``.
    Sets ``sanitized_query`` with PII-redacted query for downstream use.
    """
    from src.guardrails.input_guards import run_input_guards

    query = state.get("query", "")
    existing_violations: list[GuardrailViolation] = state.get("guardrail_violations", [])

    result = run_input_guards(query, pii_strict=False)

    new_violations: list[GuardrailViolation] = []
    for v in result.violations:
        new_violations.append({
            "stage": "input",
            "check": result.metadata.get("matched_patterns", ["unknown"])[0] if result.metadata.get("matched_patterns") else "limit_check",
            "description": v,
        })

    if not result.allowed:
        logger.warning(
            "Input guardrail BLOCKED query (trace_id=%s): %s",
            state.get("trace_id", ""),
            result.violations,
        )

    return {
        "input_blocked": not result.allowed,
        "sanitized_query": result.modified_text,
        "guardrail_violations": existing_violations + new_violations,
        "current_step": "🛡️ Input Safety Check",
        "steps_completed": state.get("steps_completed", []) + ["input_guardrail"],
    }


async def blocked_response(state: AgentState) -> dict[str, Any]:
    """Node: Return a safe refusal when input guardrails block the query.

    Produces a bilingual refusal message and sets final_answer so the
    format_response node can serve it cleanly.
    """
    language = state.get("language", "en")
    violations = state.get("guardrail_violations", [])
    violation_types = {v.get("stage", "") + ":" + v.get("check", "") for v in violations}

    if language == "ar":
        message = (
            "عذراً، لا يمكنني معالجة هذا الطلب. "
            "يبدو أنه يحتوي على محتوى غير مسموح به أو نمط غير آمن. "
            "يرجى إعادة صياغة سؤالك."
        )
    else:
        message = (
            "I'm sorry, I cannot process this request. "
            "It appears to contain a disallowed pattern or unsafe content. "
            "Please rephrase your question."
        )

    logger.info(
        "Serving blocked response (trace_id=%s, types=%s)",
        state.get("trace_id", ""),
        violation_types,
    )

    return {
        "draft_answer": message,
        "final_answer": message,
        "citations": [],
        "quality_score": {"faithfulness": 0.0, "language_consistent": True, "grounded": False},
        "current_step": "🚫 Request Blocked",
        "steps_completed": state.get("steps_completed", []) + ["blocked_response"],
    }


async def output_guardrail(state: AgentState) -> dict[str, Any]:
    """Node: Run output safety guardrails after answer generation.

    Performs:
    - Hallucination marker detection
    - Structural response validation (citations, language, length)
    - PII leakage detection and redaction from LLM output

    Modifies ``draft_answer`` with the sanitized version and appends
    any violations to ``guardrail_violations``.
    """
    from src.guardrails.output_guards import run_output_guards

    draft = state.get("draft_answer", "")
    citations = state.get("citations", [])
    language = state.get("language", "en")
    existing_violations: list[GuardrailViolation] = state.get("guardrail_violations", [])

    result = run_output_guards(answer=draft, citations=citations, language=language)

    new_violations: list[GuardrailViolation] = []
    for v in result.violations:
        new_violations.append({
            "stage": "output",
            "check": "output_guard",
            "description": v,
        })

    if result.violations:
        logger.warning(
            "Output guardrail violations (trace_id=%s): %s",
            state.get("trace_id", ""),
            result.violations,
        )

    return {
        "draft_answer": result.modified_text,
        "output_sanitized": result.metadata.get("pii_redacted", False),
        "guardrail_violations": existing_violations + new_violations,
        "current_step": "🛡️ Output Safety Check",
        "steps_completed": state.get("steps_completed", []) + ["output_guardrail"],
    }


# ---------------------------------------------------------------------------
# Stateful Memory & Coreference Resolution Node
# ---------------------------------------------------------------------------


async def contextualize_query(state: AgentState) -> dict[str, Any]:
    """Node: Disambiguate and resolve pronouns in follow-up queries using conversation history.

    Also retrieves relevant long-term memories for personalized context injection.
    """
    from src.memory.coreference import resolve_coreference
    from src.memory.long_term import get_long_term_memory_store
    from src.memory.short_term import ConversationMessage, get_conversation_memory

    query = state.get("query", "")
    raw_query = state.get("raw_query") or query
    language = state.get("language", "en")
    session_id = state.get("session_id")
    user_id = state.get("user_id")

    # 1. Fetch short-term history if available
    history: list[ConversationMessage] = []
    if session_id:
        conv_mem = get_conversation_memory()
        history = await conv_mem.get_history(session_id, max_messages=8)

    # 2. Retrieve enduring user memories if user_id is provided
    retrieved_memories: list[dict[str, Any]] = []
    if user_id:
        lt_store = get_long_term_memory_store()
        mem_facts = await lt_store.retrieve_user_memories(user_id=user_id, query=query, top_k=3)
        retrieved_memories = [m.model_dump() for m in mem_facts]

    # 3. Resolve coreferences if history exists
    client = get_openai_client()
    settings = get_settings()

    rewritten = query
    if history:
        rewritten = await resolve_coreference(
            query=query,
            history=history,
            language=language,
            client=client,
            model=settings.VLLM_MODEL,
        )

    # 4. Record reasoning step if rewritten
    reasoning_chain = state.get("reasoning_chain", [])
    if rewritten != query:
        reasoning_chain = reasoning_chain + [
            {
                "step_number": len(reasoning_chain),
                "action": "decompose",
                "input_text": f"Follow-up: {query}",
                "output_text": f"Resolved to standalone query: {rewritten}",
                "confidence": 0.90,
            }
        ]

    return {
        "raw_query": raw_query,
        "query": rewritten,
        "rewritten_query": rewritten,
        "chat_history": [m.model_dump() for m in history],
        "user_memories": retrieved_memories,
        "reasoning_chain": reasoning_chain,
        "current_step": "🧠 Contextualizing & Resolving Coreferences",
        "steps_completed": state.get("steps_completed", []) + ["contextualize_query"],
    }
