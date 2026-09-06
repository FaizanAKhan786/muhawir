import logging
from typing import Any
from src.agents.nodes import detect_language_rule_based

logger = logging.getLogger(__name__)


def detect_dominant_language(text: str) -> str:
    """Detect dominant language ('ar', 'en', or 'mixed') using rule-based character ratios."""
    lang, _ = detect_language_rule_based(text)
    return lang


def language_consistency_score(queries: list[str], responses: list[str]) -> float:
    """Calculate the fraction of responses that match the input query's language.

    For 'mixed' queries, either 'ar' or 'en' or 'mixed' is considered acceptable.
    """
    if not queries or not responses or len(queries) != len(responses):
        return 0.0

    matches = 0
    total = len(queries)

    for q, r in zip(queries, responses):
        q_lang = detect_dominant_language(q)
        r_lang = detect_dominant_language(r)

        if q_lang == "mixed":
            # For mixed queries, any valid response in Arabic or English is acceptable
            matches += 1
        elif q_lang == r_lang:
            matches += 1
        else:
            logger.debug("Language mismatch: query (%s) vs response (%s)", q_lang, r_lang)

    return round(matches / total, 4)


def code_switch_handling_score(test_cases: list[dict[str, Any]]) -> float:
    """Evaluate performance on code-switched (mixed Arabic-English) test cases.

    Each test case should have:
      - 'query': str
      - 'response': str
      - 'expected_citations': list[str] (optional)
      - 'ground_truth': str (optional)
    """
    if not test_cases:
        return 1.0

    passed = 0
    for case in test_cases:
        query = case.get("query", "")
        response = case.get("response", "")
        # Valid response should not be empty, should maintain factual terms
        if not response or len(response.strip()) < 10:
            continue

        # Check that response addresses the concepts without hallucinating gibberish
        # Check if technical terms in query are preserved in response
        query_words = set(w.lower() for w in query.split() if len(w) > 4)
        response_lower = response.lower()

        # At least one key technical word or entity retained
        overlap = any(w in response_lower for w in query_words) if query_words else True
        if overlap:
            passed += 1

    return round(passed / len(test_cases), 4)
