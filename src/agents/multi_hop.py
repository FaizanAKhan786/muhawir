"""Multi-hop reasoning engine for Muḥāwir.

Provides query complexity classification, iterative sub-query decomposition,
per-hop retrieval and synthesis, and final cross-hop answer aggregation
following a ReAct-style Think → Act → Observe loop.
"""

import json
import logging
import re
from typing import Any

from openai import AsyncOpenAI

from src.agents.state import (
    AgentState,
    Citation,
    HopContext,
    ReasoningStep,
    SubQuery,
)
from src.config import get_settings
from src.retrieval.hybrid import hybrid_retrieve
from src.retrieval.reranker import NeuralReranker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_json_from_llm(content: str, fallback: Any = None) -> Any:
    """Extract the first JSON object or array from LLM response text."""
    if not content:
        return fallback
    # Try extracting from markdown fences
    for pattern in [r"```json\s*([\s\S]*?)```", r"```\s*([\s\S]*?)```"]:
        m = re.search(pattern, content)
        if m:
            try:
                return json.loads(m.group(1).strip())
            except json.JSONDecodeError:
                continue
    # Try raw JSON array or object
    for pattern in [r"\[[\s\S]*\]", r"\{[\s\S]*\}"]:
        m = re.search(pattern, content)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                continue
    return fallback


async def _llm_call(
    client: AsyncOpenAI,
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.2,
    max_tokens: int = 500,
) -> str:
    """Make a chat completion call and return the assistant content."""
    settings = get_settings()
    model = settings.SYNTHESIS_MODEL or settings.VLLM_MODEL
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        logger.warning("LLM call failed: %s", e)
        return ""


# ---------------------------------------------------------------------------
# Node: classify_complexity
# ---------------------------------------------------------------------------

COMPLEXITY_PROMPT = """\
You are a query complexity classifier for a bilingual (Arabic/English) knowledge assistant.

Analyze the user query and determine:
1. Whether it requires multi-hop reasoning (comparing, contrasting, multi-entity, causal chains, temporal analysis).
2. A complexity score from 0.0 (trivial factoid) to 1.0 (deeply complex multi-entity reasoning).

Query: {query}
Language: {language}

Reply in JSON:
{{"is_complex": true/false, "complexity_score": 0.0-1.0, "reasoning": "brief explanation", "estimated_hops": 1-4}}"""


async def classify_complexity(state: AgentState) -> dict[str, Any]:
    """Classify whether the query needs multi-hop reasoning.

    Sets ``is_multi_hop``, ``complexity_score``, and ``max_hops`` on the state.
    Simple queries (score < COMPLEXITY_THRESHOLD) bypass multi-hop entirely.
    """
    from src.agents.nodes import get_openai_client

    query = state.get("query", "")
    language = state.get("language", "en")
    client = get_openai_client()
    settings = get_settings()

    prompt = COMPLEXITY_PROMPT.format(query=query, language=language)
    content = await _llm_call(client, [{"role": "user", "content": prompt}], max_tokens=200)
    parsed = _parse_json_from_llm(content, {})

    is_complex = bool(parsed.get("is_complex", False))
    score = float(parsed.get("complexity_score", 0.3))
    estimated_hops = int(parsed.get("estimated_hops", 1))

    # Override: if score below threshold, force single-hop
    if score < settings.COMPLEXITY_THRESHOLD:
        is_complex = False
        estimated_hops = 1

    max_hops = min(estimated_hops, settings.MAX_HOPS)

    reasoning_step: ReasoningStep = {
        "step_number": 0,
        "action": "decompose",
        "input_text": query,
        "output_text": f"Complexity={score:.2f}, multi_hop={is_complex}, hops={max_hops}",
        "confidence": score,
    }

    return {
        "is_multi_hop": is_complex,
        "complexity_score": score,
        "max_hops": max_hops,
        "current_hop": 0,
        "sub_queries": [],
        "reasoning_chain": [reasoning_step],
        "hop_contexts": [],
        "synthesis_answer": "",
    }


# ---------------------------------------------------------------------------
# Node: decompose_query
# ---------------------------------------------------------------------------

DECOMPOSE_PROMPT = """\
You are a query decomposition expert for a bilingual (Arabic/English) knowledge assistant.

Break this complex query into 2-{max_sub} simpler, self-contained sub-queries that can each be answered independently. Each sub-query should target a specific piece of information needed.

Original query: {query}
Language: {language}
Previously gathered context (if any): {prior_context}

Reply in JSON array format:
[
  {{"sub_query": "...", "purpose": "why this sub-query is needed"}},
  ...
]"""


async def decompose_query(state: AgentState) -> dict[str, Any]:
    """Decompose a complex query into independent sub-queries.

    Uses prior hop context (if available) to avoid re-asking answered questions
    and to generate follow-up sub-queries based on partial findings.
    """
    from src.agents.nodes import get_openai_client

    query = state.get("query", "")
    language = state.get("language", "en")
    client = get_openai_client()
    settings = get_settings()
    hop_contexts = state.get("hop_contexts", [])

    # Build prior context summary from previous hops
    prior_parts: list[str] = []
    for hc in hop_contexts:
        if hc.get("partial_answer"):
            prior_parts.append(f"Hop {hc['hop_number']}: {hc['partial_answer'][:200]}")
    prior_context = " | ".join(prior_parts) if prior_parts else "None yet."

    prompt = DECOMPOSE_PROMPT.format(
        query=query,
        language=language,
        max_sub=settings.SUB_QUERY_MAX,
        prior_context=prior_context,
    )

    content = await _llm_call(client, [{"role": "user", "content": prompt}], max_tokens=400)
    parsed = _parse_json_from_llm(content, [])

    sub_queries: list[SubQuery] = []
    current_hop = state.get("current_hop", 0) + 1

    if isinstance(parsed, list):
        for item in parsed[: settings.SUB_QUERY_MAX]:
            sq: SubQuery = {
                "sub_query": item.get("sub_query", ""),
                "purpose": item.get("purpose", ""),
                "answered": False,
                "answer": "",
                "supporting_docs": [],
                "hop_number": current_hop,
            }
            if sq["sub_query"]:
                sub_queries.append(sq)

    # Fallback: if decomposition failed, use original query
    if not sub_queries:
        sub_queries = [
            {
                "sub_query": query,
                "purpose": "Direct retrieval (decomposition fallback)",
                "answered": False,
                "answer": "",
                "supporting_docs": [],
                "hop_number": current_hop,
            }
        ]

    reasoning_step: ReasoningStep = {
        "step_number": len(state.get("reasoning_chain", [])),
        "action": "decompose",
        "input_text": query,
        "output_text": f"Decomposed into {len(sub_queries)} sub-queries for hop {current_hop}",
        "confidence": 0.8,
    }

    return {
        "sub_queries": sub_queries,
        "current_hop": current_hop,
        "reasoning_chain": state.get("reasoning_chain", []) + [reasoning_step],
    }


# ---------------------------------------------------------------------------
# Node: multi_hop_retrieve
# ---------------------------------------------------------------------------

async def multi_hop_retrieve(state: AgentState) -> dict[str, Any]:
    """Execute retrieval for each unanswered sub-query in the current hop.

    Runs all sub-query retrievals concurrently, stores per-sub-query docs,
    and aggregates into ``retrieved_docs`` for downstream reranking.
    """
    import asyncio

    sub_queries = state.get("sub_queries", [])
    language = state.get("language", "en")
    all_docs = []
    updated_subs: list[SubQuery] = []

    async def _retrieve_for_sub(sq: SubQuery) -> SubQuery:
        if sq.get("answered"):
            return sq
        docs = await hybrid_retrieve(query=sq["sub_query"], language=language, top_k=15)
        return {
            **sq,
            "supporting_docs": docs,
        }

    # Concurrent retrieval for all sub-queries
    tasks = [_retrieve_for_sub(sq) for sq in sub_queries]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in results:
        if isinstance(result, Exception):
            logger.warning("Sub-query retrieval failed: %s", result)
            continue
        updated_subs.append(result)
        all_docs.extend(result.get("supporting_docs", []))

    # Deduplicate by chunk_id
    seen_chunks: set[str] = set()
    unique_docs = []
    for doc in all_docs:
        cid = doc.get("chunk_id", "")
        if cid not in seen_chunks:
            seen_chunks.add(cid)
            unique_docs.append(doc)

    reasoning_step: ReasoningStep = {
        "step_number": len(state.get("reasoning_chain", [])),
        "action": "retrieve",
        "input_text": f"{len(updated_subs)} sub-queries",
        "output_text": f"Retrieved {len(unique_docs)} unique chunks across {len(updated_subs)} sub-queries",
        "confidence": 0.85,
    }

    return {
        "sub_queries": updated_subs,
        "retrieved_docs": unique_docs,
        "reasoning_chain": state.get("reasoning_chain", []) + [reasoning_step],
    }


# ---------------------------------------------------------------------------
# Node: multi_hop_rerank
# ---------------------------------------------------------------------------

async def multi_hop_rerank(state: AgentState) -> dict[str, Any]:
    """Rerank retrieved documents per sub-query and globally.

    Each sub-query gets its own reranked top-K, then all are merged
    into a global reranked set for the synthesizer.
    """
    from src.agents.nodes import get_reranker

    sub_queries = state.get("sub_queries", [])
    retrieved_docs = state.get("retrieved_docs", [])
    query = state.get("query", "")
    reranker = get_reranker()

    # Per-sub-query reranking
    updated_subs: list[SubQuery] = []
    all_reranked = []

    for sq in sub_queries:
        sq_docs = sq.get("supporting_docs", [])
        if sq_docs:
            reranked = reranker.rerank(query=sq["sub_query"], documents=sq_docs, top_k=5)
            updated_subs.append({**sq, "supporting_docs": reranked})
            all_reranked.extend(reranked)
        else:
            updated_subs.append(sq)

    # Global reranking against the original query
    if all_reranked:
        # Deduplicate before global rerank
        seen: set[str] = set()
        unique_reranked = []
        for doc in all_reranked:
            cid = doc.get("chunk_id", "")
            if cid not in seen:
                seen.add(cid)
                unique_reranked.append(doc)
        global_reranked = reranker.rerank(query=query, documents=unique_reranked, top_k=10)
    else:
        global_reranked = reranker.rerank(query=query, documents=retrieved_docs, top_k=5)

    return {
        "sub_queries": updated_subs,
        "reranked_docs": global_reranked,
    }


# ---------------------------------------------------------------------------
# Node: hop_synthesize
# ---------------------------------------------------------------------------

HOP_SYNTHESIS_PROMPT = """\
You are a multi-hop reasoning synthesizer for a bilingual knowledge assistant.

You are on hop {hop_number} of {max_hops}.

Original query: {query}
Language: {language}

Sub-queries and their retrieved context:
{sub_query_context}

Previous hop findings:
{prior_findings}

For each sub-query, provide a concise answer based ONLY on the retrieved context.
Then assess: do you have enough information to fully answer the original query?

Reply in JSON:
{{
  "sub_answers": [
    {{"sub_query": "...", "answer": "...", "confidence": 0.0-1.0, "needs_more_info": true/false}}
  ],
  "partial_synthesis": "Combined understanding so far...",
  "needs_another_hop": true/false,
  "next_questions": ["follow-up question if needed", ...]
}}"""


async def hop_synthesize(state: AgentState) -> dict[str, Any]:
    """Synthesize answers for the current hop's sub-queries.

    Determines if another hop is needed (information gaps remain) or if
    we have enough context to produce a final synthesis.
    """
    from src.agents.nodes import get_openai_client

    query = state.get("query", "")
    language = state.get("language", "en")
    client = get_openai_client()
    settings = get_settings()
    sub_queries = state.get("sub_queries", [])
    hop_contexts = state.get("hop_contexts", [])
    current_hop = state.get("current_hop", 1)
    max_hops = state.get("max_hops", 3)

    # Build sub-query context
    sq_context_parts = []
    for i, sq in enumerate(sub_queries, 1):
        docs_text = "\n".join(
            [f"  - {d['text'][:200]}" for d in sq.get("supporting_docs", [])[:3]]
        )
        sq_context_parts.append(
            f"Sub-query {i}: {sq['sub_query']}\nPurpose: {sq['purpose']}\nContext:\n{docs_text}"
        )
    sq_context = "\n\n".join(sq_context_parts) if sq_context_parts else "No sub-queries."

    # Build prior findings
    prior_parts = []
    for hc in hop_contexts:
        prior_parts.append(f"Hop {hc['hop_number']}: {hc.get('partial_answer', 'No answer')[:300]}")
    prior_findings = "\n".join(prior_parts) if prior_parts else "First hop."

    prompt = HOP_SYNTHESIS_PROMPT.format(
        hop_number=current_hop,
        max_hops=max_hops,
        query=query,
        language=language,
        sub_query_context=sq_context,
        prior_findings=prior_findings,
    )

    content = await _llm_call(client, [{"role": "user", "content": prompt}], max_tokens=800)
    parsed = _parse_json_from_llm(content, {})

    # Update sub-queries with answers
    sub_answers = parsed.get("sub_answers", [])
    updated_subs: list[SubQuery] = []
    avg_confidence = 0.0

    for sq in sub_queries:
        matched_answer = next(
            (sa for sa in sub_answers if sa.get("sub_query", "").strip() == sq["sub_query"].strip()),
            None,
        )
        if matched_answer:
            updated_subs.append({
                **sq,
                "answered": True,
                "answer": matched_answer.get("answer", ""),
            })
            avg_confidence += float(matched_answer.get("confidence", 0.5))
        else:
            updated_subs.append({**sq, "answered": False})
            avg_confidence += 0.3

    if updated_subs:
        avg_confidence /= len(updated_subs)

    partial_answer = parsed.get("partial_synthesis", "")
    needs_more = bool(parsed.get("needs_another_hop", False))
    next_questions = parsed.get("next_questions", [])

    # Build hop context record
    hop_ctx: HopContext = {
        "hop_number": current_hop,
        "query": query,
        "retrieved_docs": state.get("retrieved_docs", []),
        "reranked_docs": state.get("reranked_docs", []),
        "partial_answer": partial_answer,
        "confidence": avg_confidence,
    }

    reasoning_step: ReasoningStep = {
        "step_number": len(state.get("reasoning_chain", [])),
        "action": "synthesize",
        "input_text": f"Hop {current_hop}: {len(updated_subs)} sub-queries",
        "output_text": f"Confidence={avg_confidence:.2f}, needs_more={needs_more}",
        "confidence": avg_confidence,
    }

    # If another hop is needed and we haven't hit max, prepare follow-up sub-queries
    new_sub_queries: list[SubQuery] = []
    if needs_more and current_hop < max_hops and next_questions:
        for nq in next_questions[: settings.SUB_QUERY_MAX]:
            new_sub_queries.append({
                "sub_query": nq,
                "purpose": "Follow-up from previous hop",
                "answered": False,
                "answer": "",
                "supporting_docs": [],
                "hop_number": current_hop + 1,
            })

    result: dict[str, Any] = {
        "sub_queries": new_sub_queries if new_sub_queries else updated_subs,
        "hop_contexts": hop_contexts + [hop_ctx],
        "synthesis_answer": partial_answer,
        "reasoning_chain": state.get("reasoning_chain", []) + [reasoning_step],
    }

    # Store whether we should continue hopping
    result["_needs_another_hop"] = bool(new_sub_queries) and current_hop < max_hops

    return result


# ---------------------------------------------------------------------------
# Node: final_synthesize
# ---------------------------------------------------------------------------

FINAL_SYNTHESIS_PROMPT = """\
You are Muḥāwir (محاور), a bilingual knowledge assistant performing multi-hop reasoning synthesis.

Original query: {query}
Language: {language}

Reasoning chain across {num_hops} hops:
{hop_summaries}

All supporting evidence:
{evidence}

Produce a comprehensive final answer that:
1. Answers in the SAME language as the user's question.
2. Integrates findings from all reasoning hops.
3. Cites sources: [Source 1], [Source 2].
4. If information is incomplete, acknowledges gaps.
5. Preserves technical terms in original language.

Respond in JSON:
{{"answer": "...", "language": "ar|en", "citations": [{{"doc_id": "...", "chunk_id": "...", "text_snippet": "..."}}], "confidence": 0.0-1.0, "reasoning_summary": "brief chain-of-thought"}}"""


async def final_synthesize(state: AgentState) -> dict[str, Any]:
    """Produce the final answer by synthesizing across all hop contexts.

    Merges partial answers, evidence, and reasoning chain into a single
    coherent response with citations and confidence score.
    """
    from src.agents.nodes import get_openai_client

    query = state.get("query", "")
    language = state.get("language", "en")
    client = get_openai_client()
    hop_contexts = state.get("hop_contexts", [])
    reranked_docs = state.get("reranked_docs", [])

    # Build hop summaries
    hop_parts = []
    for hc in hop_contexts:
        hop_parts.append(
            f"Hop {hc['hop_number']} (confidence={hc['confidence']:.2f}): {hc.get('partial_answer', 'N/A')[:300]}"
        )
    hop_summaries = "\n".join(hop_parts) if hop_parts else "Single-hop reasoning."

    # Build evidence
    evidence_parts = []
    all_docs = reranked_docs
    for hc in hop_contexts:
        all_docs = all_docs + hc.get("reranked_docs", [])

    # Deduplicate
    seen: set[str] = set()
    for idx, doc in enumerate(all_docs):
        cid = doc.get("chunk_id", "")
        if cid not in seen:
            seen.add(cid)
            evidence_parts.append(
                f"[Source {len(evidence_parts)+1}] (doc_id: {doc['doc_id']}, chunk_id: {cid}): {doc['text'][:250]}"
            )
        if len(evidence_parts) >= 10:
            break

    evidence = "\n\n".join(evidence_parts) if evidence_parts else "No evidence gathered."

    prompt = FINAL_SYNTHESIS_PROMPT.format(
        query=query,
        language=language,
        num_hops=len(hop_contexts),
        hop_summaries=hop_summaries,
        evidence=evidence,
    )

    content = await _llm_call(
        client,
        [
            {"role": "system", "content": "You are a multi-hop reasoning synthesizer."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=1000,
    )
    parsed = _parse_json_from_llm(content, {})

    draft_answer = parsed.get("answer", state.get("synthesis_answer", ""))
    citations: list[Citation] = []
    for cit in parsed.get("citations", []):
        citations.append({
            "doc_id": cit.get("doc_id", ""),
            "chunk_id": cit.get("chunk_id", ""),
            "text_snippet": cit.get("text_snippet", ""),
            "source": cit.get("source", ""),
        })

    reasoning_step: ReasoningStep = {
        "step_number": len(state.get("reasoning_chain", [])),
        "action": "synthesize",
        "input_text": f"Final synthesis across {len(hop_contexts)} hops",
        "output_text": f"Answer length={len(draft_answer)}, citations={len(citations)}",
        "confidence": float(parsed.get("confidence", 0.8)),
    }

    return {
        "draft_answer": draft_answer,
        "citations": citations,
        "reasoning_chain": state.get("reasoning_chain", []) + [reasoning_step],
    }
