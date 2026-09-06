import logging
from typing import Any
from src.agents.state import RetrievedDocument

logger = logging.getLogger(__name__)


class NeuralReranker:
    """Neural cross-encoder reranker based on BAAI/bge-reranker-v2-m3."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", cross_encoder: Any = None):
        self.model_name = model_name
        self._cross_encoder = cross_encoder

    @property
    def model(self):
        if self._cross_encoder is None:
            from sentence_transformers import CrossEncoder

            logger.info("Loading CrossEncoder reranker model: %s", self.model_name)
            self._cross_encoder = CrossEncoder(self.model_name)
        return self._cross_encoder

    def rerank(
        self,
        query: str,
        documents: list[RetrievedDocument],
        top_k: int = 5,
    ) -> list[RetrievedDocument]:
        """Rerank retrieved documents using cross-encoder scores."""
        if not documents:
            return []

        if len(documents) <= 1:
            return documents[:top_k]

        pairs = [[query, doc["text"]] for doc in documents]
        try:
            scores = self.model.predict(pairs)
        except Exception as e:
            logger.warning("CrossEncoder prediction failed, falling back to original scores: %s", e)
            return sorted(documents, key=lambda d: d.get("score", 0.0), reverse=True)[:top_k]

        scored_docs: list[RetrievedDocument] = []
        for doc, score in zip(documents, scores):
            updated_doc = dict(doc)
            updated_doc["score"] = float(score)
            scored_docs.append(updated_doc)  # type: ignore

        scored_docs.sort(key=lambda d: d["score"], reverse=True)
        return scored_docs[:top_k]
