import asyncio
import logging
import re
from typing import Any
from rank_bm25 import BM25Okapi

from src.agents.state import RetrievedDocument
from src.retrieval.chunking import normalize_arabic, normalize_english

logger = logging.getLogger(__name__)


def tokenize_for_bm25(text: str, language: str = "auto") -> list[str]:
    """Tokenize query or document for BM25 with language-aware normalization."""
    # If contains Arabic characters or explicitly Arabic, normalize Arabic
    if any("\u0600" <= c <= "\u06FF" for c in text) or language == "ar":
        text = normalize_arabic(text)
    else:
        text = normalize_english(text)

    tokens = re.findall(r"\w+", text)
    return tokens if tokens else [text]


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Merge ranked lists using Reciprocal Rank Fusion (RRF).

    For each document, RRF_score = sum(1 / (k + rank_i)) across all lists
    where rank_i is 1-indexed.
    """
    scores: dict[str, float] = {}

    for ranked_list in ranked_lists:
        for rank, (doc_id, _) in enumerate(ranked_list, start=1):
            if doc_id not in scores:
                scores[doc_id] = 0.0
            scores[doc_id] += 1.0 / (k + rank)

    # Sort descending by RRF score
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return sorted_scores


class BM25Index:
    """In-memory BM25 index over chunked documents."""

    def __init__(self, corpus: list[dict[str, Any]] | None = None):
        self.doc_map: dict[str, dict[str, Any]] = {}
        self.doc_ids: list[str] = []
        self.bm25: BM25Okapi | None = None

        if corpus:
            self.build_index(corpus)

    def build_index(self, corpus: list[dict[str, Any]]) -> None:
        self.doc_map.clear()
        self.doc_ids.clear()
        tokenized_corpus = []

        for item in corpus:
            # Use chunk_id if present, else doc_id
            cid = item.get("chunk_id") or item.get("doc_id") or f"doc_{len(self.doc_ids)}"
            text = item.get("text", "")
            lang = item.get("language", "en")

            self.doc_ids.append(cid)
            self.doc_map[cid] = item
            tokenized_corpus.append(tokenize_for_bm25(text, language=lang))

        if tokenized_corpus:
            self.bm25 = BM25Okapi(tokenized_corpus)
        else:
            self.bm25 = None

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """Search BM25 index and return list of (doc_id/chunk_id, score) pairs."""
        if not self.bm25 or not self.doc_ids:
            return []

        tokens = tokenize_for_bm25(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results: list[tuple[str, float]] = []
        for idx in ranked_indices[:top_k]:
            score = float(scores[idx])
            if score > 0.0 or len(results) < top_k:
                results.append((self.doc_ids[idx], score))

        return results


class VectorSearcher:
    """Vector searcher querying Qdrant using dense BGE-m3 embeddings."""

    def __init__(
        self,
        qdrant_client: Any,
        collection_name: str,
        embedding_model: Any = None,
    ):
        self.client = qdrant_client
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.doc_map: dict[str, dict[str, Any]] = {}

    def _embed_query(self, query: str) -> list[float]:
        if self.embedding_model is not None:
            if hasattr(self.embedding_model, "encode"):
                emb = self.embedding_model.encode(query)
                if hasattr(emb, "tolist"):
                    return emb.tolist()
                return list(emb)
        from src.retrieval.chunking import simple_hash_vector
        return simple_hash_vector(query, dim=1024)

    async def search(
        self,
        query: str,
        top_k: int = 20,
        language_filter: str | None = None,
    ) -> list[tuple[str, float]]:
        """Embed query and search Qdrant collection with optional language filter."""
        query_vector = self._embed_query(query)

        filter_condition = None
        if language_filter and language_filter not in ("auto", "mixed"):
            try:
                from qdrant_client.http import models as rest_models
                filter_condition = rest_models.Filter(
                    must=[
                        rest_models.FieldCondition(
                            key="language",
                            match=rest_models.MatchValue(value=language_filter),
                        )
                    ]
                )
            except Exception:
                filter_condition = None

        results: list[tuple[str, float]] = []
        try:
            if self.client is not None:
                # Support both async client and sync client
                search_call = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    query_filter=filter_condition,
                    limit=top_k,
                )
                hits = await search_call if hasattr(search_call, "__await__") else search_call
                for hit in hits:
                    cid = str(hit.id)
                    score = float(hit.score)
                    if hasattr(hit, "payload") and hit.payload:
                        self.doc_map[cid] = hit.payload
                    results.append((cid, score))
        except Exception as e:
            logger.warning("Vector search error in Qdrant: %s", e)

        return results


# Global registries for retrieval components
_GLOBAL_BM25_INDEX: BM25Index | None = None
_GLOBAL_VECTOR_SEARCHER: VectorSearcher | None = None
_GLOBAL_DOCUMENT_STORE: dict[str, dict[str, Any]] = {}


def set_retrieval_components(
    bm25: BM25Index | None = None,
    vector_searcher: VectorSearcher | None = None,
    document_store: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Register global retrieval components."""
    global _GLOBAL_BM25_INDEX, _GLOBAL_VECTOR_SEARCHER, _GLOBAL_DOCUMENT_STORE
    if bm25 is not None:
        _GLOBAL_BM25_INDEX = bm25
    if vector_searcher is not None:
        _GLOBAL_VECTOR_SEARCHER = vector_searcher
    if document_store is not None:
        _GLOBAL_DOCUMENT_STORE.update(document_store)


def get_retrieval_components():
    return _GLOBAL_BM25_INDEX, _GLOBAL_VECTOR_SEARCHER, _GLOBAL_DOCUMENT_STORE


async def hybrid_retrieve(
    query: str,
    language: str = "auto",
    top_k: int = 20,
    bm25_index: BM25Index | None = None,
    vector_searcher: VectorSearcher | None = None,
) -> list[RetrievedDocument]:
    """Execute parallel BM25 and vector search, merging results with RRF."""
    bm25 = bm25_index or _GLOBAL_BM25_INDEX
    vec = vector_searcher or _GLOBAL_VECTOR_SEARCHER
    global _GLOBAL_DOCUMENT_STORE

    # Run BM25 and Vector search in PARALLEL via asyncio.gather
    async def _run_bm25():
        if bm25 is None:
            return []
        return bm25.search(query, top_k=top_k)

    async def _run_vec():
        if vec is None:
            return []
        lang_filter = language if language in ("ar", "en") else None
        return await vec.search(query, top_k=top_k, language_filter=lang_filter)

    bm25_results, vec_results = await asyncio.gather(_run_bm25(), _run_vec())

    # Merge ranked lists via RRF
    rrf_fused = reciprocal_rank_fusion([bm25_results, vec_results], k=60)
    top_fused = rrf_fused[:top_k]

    # Reconstruct RetrievedDocument objects
    retrieved_docs: list[RetrievedDocument] = []
    for doc_id, score in top_fused:
        doc_data = None
        if bm25 and doc_id in bm25.doc_map:
            doc_data = bm25.doc_map[doc_id]
        elif vec and doc_id in vec.doc_map:
            doc_data = vec.doc_map[doc_id]
        elif doc_id in _GLOBAL_DOCUMENT_STORE:
            doc_data = _GLOBAL_DOCUMENT_STORE[doc_id]

        if doc_data:
            retrieved_docs.append(
                {
                    "doc_id": doc_data.get("doc_id", doc_id),
                    "chunk_id": doc_data.get("chunk_id", doc_id),
                    "text": doc_data.get("text", ""),
                    "source": doc_data.get("metadata", {}).get("source", doc_data.get("source", doc_id)),
                    "score": float(score),
                    "language": doc_data.get("language", language if language != "auto" else "en"),
                    "metadata": doc_data.get("metadata", {}),
                }
            )
        else:
            # Fallback document representation if metadata was stripped
            retrieved_docs.append(
                {
                    "doc_id": doc_id,
                    "chunk_id": doc_id,
                    "text": f"Document content for {doc_id}",
                    "source": doc_id,
                    "score": float(score),
                    "language": language if language != "auto" else "en",
                    "metadata": {},
                }
            )

    return retrieved_docs
