import json
import logging
from typing import Any
import numpy as np

logger = logging.getLogger(__name__)


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Calculate cosine similarity between two float vectors."""
    a = np.array(v1, dtype=np.float32)
    b = np.array(v2, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class SemanticCache:
    """Redis-backed semantic cache with vector similarity lookup."""

    KEY_PREFIX = "muhawir:cache:"
    INDEX_SET = "muhawir:cache_keys"

    def __init__(
        self,
        redis_client: Any,
        embedding_model: Any = None,
        threshold: float = 0.95,
    ):
        self.redis = redis_client
        self._embedding_model = embedding_model
        self.threshold = threshold

    def _get_embedding(self, text: str) -> list[float]:
        if self._embedding_model is not None:
            if hasattr(self._embedding_model, "encode"):
                emb = self._embedding_model.encode(text)
                if hasattr(emb, "tolist"):
                    return emb.tolist()
                return list(emb)
        # Fallback simple deterministic pseudo-embedding for testing or when model not passed
        from src.retrieval.chunking import simple_hash_vector
        return simple_hash_vector(text, dim=128)

    async def get(self, query: str) -> dict[str, Any] | None:
        """Find cached response if query cosine similarity >= threshold."""
        if not self.redis:
            return None

        try:
            query_emb = self._get_embedding(query)
            keys = await self._get_all_keys()
            if not keys:
                return None

            best_sim = -1.0
            best_response = None

            for key in keys:
                raw_data = await self._get_key_data(key)
                if not raw_data:
                    continue
                data = json.loads(raw_data)
                cached_emb = data.get("embedding")
                if not cached_emb:
                    continue

                sim = cosine_similarity(query_emb, cached_emb)
                if sim > best_sim:
                    best_sim = sim
                    best_response = data.get("response")

            if best_sim >= self.threshold and best_response is not None:
                logger.info("Semantic cache HIT (similarity: %.4f)", best_sim)
                return best_response

            return None
        except Exception as e:
            logger.warning("Error reading from semantic cache: %s", e)
            return None

    async def set(self, query: str, response: dict[str, Any], ttl: int = 3600) -> None:
        """Store query embedding and response in Redis."""
        if not self.redis:
            return

        try:
            query_emb = self._get_embedding(query)
            import hashlib

            key_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]
            key = f"{self.KEY_PREFIX}{key_hash}"
            payload = json.dumps(
                {
                    "query": query,
                    "embedding": query_emb,
                    "response": response,
                }
            )

            # Support both async and sync redis clients / mocks
            if hasattr(self.redis, "setex"):
                res = self.redis.setex(key, ttl, payload)
                if hasattr(res, "__await__"):
                    await res
            elif hasattr(self.redis, "set"):
                res = self.redis.set(key, payload, ex=ttl)
                if hasattr(res, "__await__"):
                    await res

            if hasattr(self.redis, "sadd"):
                res2 = self.redis.sadd(self.INDEX_SET, key)
                if hasattr(res2, "__await__"):
                    await res2
        except Exception as e:
            logger.warning("Error writing to semantic cache: %s", e)

    async def invalidate_all(self) -> None:
        """Flush semantic cache."""
        if not self.redis:
            return
        try:
            keys = await self._get_all_keys()
            for key in keys:
                if hasattr(self.redis, "delete"):
                    res = self.redis.delete(key)
                    if hasattr(res, "__await__"):
                        await res
            if hasattr(self.redis, "delete"):
                res = self.redis.delete(self.INDEX_SET)
                if hasattr(res, "__await__"):
                    await res
        except Exception as e:
            logger.warning("Error invalidating semantic cache: %s", e)

    async def _get_all_keys(self) -> list[str]:
        if hasattr(self.redis, "smembers"):
            res = self.redis.smembers(self.INDEX_SET)
            if hasattr(res, "__await__"):
                members = await res
            else:
                members = res
            return [m.decode("utf-8") if isinstance(m, bytes) else str(m) for m in members]
        return []

    async def _get_key_data(self, key: str) -> str | None:
        if hasattr(self.redis, "get"):
            res = self.redis.get(key)
            if hasattr(res, "__await__"):
                val = await res
            else:
                val = res
            if val is None:
                return None
            return val.decode("utf-8") if isinstance(val, bytes) else str(val)
        return None
