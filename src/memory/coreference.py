"""Bilingual Coreference Resolution and Query Disambiguation.

Rewrites anaphoric and elliptical follow-up questions (e.g. "What about its second pillar?",
"وما هي ركائزها؟") into standalone, context-independent search queries using dialogue history.
"""

import json
import logging
import re
from typing import Any

from src.memory.short_term import ConversationMessage

logger = logging.getLogger(__name__)

# Pronouns and anaphoric markers indicating dependency on chat history
_ANAPHORA_PATTERNS = [
    # English anaphoric references
    r"\b(it|its|they|their|them|this|that|these|those)\b",
    r"\b(the\s+former|the\s+latter|the\s+second|the\s+first|the\s+other|both)\b",
    r"\b(what\s+about|how\s+about|and\s+its|and\s+their|compared\s+to|same\s+for)\b",
    r"\b(why\s+is\s+that|how\s+so|tell\s+me\s+more|elaborate|continue)\b",
    # Arabic anaphoric references (أسماء إشارة وضمائر تابعة وتراكيب المتابعة)
    r"\b(ذلك|تلك|هذا|هذه|هؤلاء|هذان|هاتان|كلاهما|بينهما|السابق|الأخير)\b",
    r"\b(ماذا\s+عن|وكيف\s+عن|وماذا\s+عنه|وماذا\s+عنها|مقارنة\s+بها|قارن\s+بينهما)\b",
    r"\b(ركائزها|أهدافها|خطتها|محاورها|ميزانيتها|تاريخها|أهميتها|نتائجها|تأثيرها)\b",
]


def is_contextual_query(query: str, language: str = "en") -> bool:
    """Fast heuristic check whether a query likely depends on conversation history."""
    clean_q = query.strip().lower()

    # If the query contains an explicit self-contained entity subject and no clear follow-up pronoun, it's standalone
    standalone_subjects = [
        "رؤية المملكة العربية السعودية",
        "رؤية السعودية 2030",
        "رؤية قطر الوطنية 2030",
        "qatar national vision 2030",
        "saudi vision 2030",
    ]
    for subj in standalone_subjects:
        if subj in clean_q:
            # Check if it has an explicit contextual follow-up modifier (e.g. "compared to its", "what about")
            if not any(k in clean_q for k in ["what about", "how about", "compared to", "مقارنة", "ماذا عن"]):
                return False

    # Very short queries (1-4 words) in a continuing dialogue are almost always contextual
    words = clean_q.split()
    if len(words) <= 4 and not any(k in clean_q for k in ["رؤية", "vision", "qatar", "saudi"]):
        return True

    # Strip standard interrogative copula "ما هو / ما هي" at start before checking pronouns
    check_text = re.sub(r"^(ما\s+هو|ما\s+هي|ما\s+هما|ما\s+هم|ما\s+هن)\s+", "", clean_q)

    # Check for anaphoric pronouns and indicators
    for pat in _ANAPHORA_PATTERNS:
        if re.search(pat, check_text, re.IGNORECASE | re.UNICODE):
            return True

    return False


COREFERENCE_PROMPT_TEMPLATE = """\
You are an expert bilingual (Arabic/English) query rewriter.

Given a conversation history between a User and an Assistant, and the User's latest follow-up question, rewrite the question into a completely SELF-CONTAINED search query.
Resolve all pronouns (e.g. "it", "its", "they", "ها", "هو", "ذلك"), references to previous topics, or ellipsis.

Rules:
1. Preserve the EXACT language of the user's latest query (Arabic if Arabic, English if English).
2. Do NOT answer the question. Only rewrite it.
3. If the latest question is already completely self-contained, return it unchanged.
4. Output ONLY JSON: {{"rewritten_query": "...", "is_rewritten": true/false, "referenced_topic": "..."}}

Conversation History:
{history}

User's Latest Query:
{query}"""


async def resolve_coreference(
    query: str,
    history: list[ConversationMessage],
    language: str = "en",
    client: Any = None,
    model: str = "Qwen/Qwen2.5-14B-Instruct",
) -> str:
    """Resolve pronouns and contextual references in the query using recent history.

    Args:
        query: The raw user query.
        history: List of recent ConversationMessages.
        language: Detected language code ("ar", "en", "mixed").
        client: AsyncOpenAI client instance.
        model: Model name for LLM call.

    Returns:
        The standalone rewritten query (or original query if no rewrite needed/possible).
    """
    if not history or len(history) == 0:
        return query

    # Only run LLM rewrite if heuristic suggests anaphora/context dependence
    if not is_contextual_query(query, language=language):
        logger.debug("Query appears self-contained, skipping coreference rewrite: %s", query)
        return query

    if client is None:
        logger.debug("No LLM client provided for coreference resolution, returning original query.")
        return query

    # Format last 4 messages of history
    history_lines = []
    for m in history[-4:]:
        role = "User" if m.role == "user" else "Assistant"
        snippet = m.content[:200]
        history_lines.append(f"{role}: {snippet}")
    formatted_history = "\n".join(history_lines)

    prompt = COREFERENCE_PROMPT_TEMPLATE.format(
        history=formatted_history,
        query=query,
    )

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
        )
        content = response.choices[0].message.content or "{}"
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group(0))
            rewritten = parsed.get("rewritten_query", "").strip()
            if rewritten and len(rewritten) > 3:
                logger.info(
                    "Coreference resolved: '%s' -> '%s' (topic: %s)",
                    query,
                    rewritten,
                    parsed.get("referenced_topic", "n/a"),
                )
                return rewritten
    except Exception as e:
        logger.warning("Coreference resolution failed: %s", e)

    return query
