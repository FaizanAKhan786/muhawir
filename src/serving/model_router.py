from dataclasses import dataclass
from enum import Enum
import re
from src.config import get_settings


class QueryComplexity(str, Enum):
    SIMPLE = "simple"
    COMPLEX = "complex"


@dataclass
class RouteDecision:
    model_name: str
    complexity: QueryComplexity
    reason: str


class ModelRouter:
    """Routes queries to lightweight models for simple lookups or full models for complex synthesis."""

    def __init__(
        self,
        full_model: str | None = None,
        small_model: str = "Qwen/Qwen2.5-7B-Instruct",
    ):
        settings = get_settings()
        self.full_model = full_model or settings.VLLM_MODEL
        self.small_model = small_model

        # Complex indicators
        self.complex_keywords = [
            "compare",
            "contrast",
            "synthesize",
            "analyze",
            "explain in detail",
            "step by step",
            "قارن",
            "حلل",
            "اشرح بالتفصيل",
            "لخص وقارن",
            "ما هي الفروقات",
            "ما الفرق بين",
        ]

    def assess_complexity(self, query: str) -> QueryComplexity:
        """Determine if a query is simple or complex."""
        clean_q = query.lower().strip()

        # Check length
        words = clean_q.split()
        if len(words) > 25:
            return QueryComplexity.COMPLEX

        # Check multi-question or clause indicators
        if "?" in clean_q and clean_q.count("?") > 1:
            return QueryComplexity.COMPLEX
        if "؟" in clean_q and clean_q.count("؟") > 1:
            return QueryComplexity.COMPLEX

        # Check complex analytical keywords
        for kw in self.complex_keywords:
            if kw in clean_q:
                return QueryComplexity.COMPLEX

        # Check if code switching is heavy
        arabic_chars = len(re.findall(r"[\u0600-\u06FF]", query))
        english_chars = len(re.findall(r"[A-Za-z]", query))
        if arabic_chars > 10 and english_chars > 10:
            return QueryComplexity.COMPLEX

        return QueryComplexity.SIMPLE

    def route(self, query: str) -> RouteDecision:
        """Make a model routing decision."""
        complexity = self.assess_complexity(query)
        if complexity == QueryComplexity.COMPLEX:
            return RouteDecision(
                model_name=self.full_model,
                complexity=complexity,
                reason="Query contains complex keywords, multiple clauses, or requires multi-document synthesis.",
            )
        else:
            return RouteDecision(
                model_name=self.small_model,
                complexity=complexity,
                reason="Query is short, factual, or straightforward.",
            )


_ROUTER: ModelRouter | None = None


def route_model(query: str) -> RouteDecision:
    """Singleton helper function for query routing."""
    global _ROUTER
    if _ROUTER is None:
        _ROUTER = ModelRouter()
    return _ROUTER.route(query)
