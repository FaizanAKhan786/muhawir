"""Long-term semantic memory store for user preferences, topic interests, and profile facts.

Persists user facts across sessions in Qdrant (or local fallback) and retrieves them
semantically to personalize agent responses and maintain continuity.
"""

from datetime import datetime, timezone
import json
import logging
import re
import uuid
from typing import Any, Literal
from pydantic import BaseModel, Field

from src.retrieval.chunking import simple_hash_vector

logger = logging.getLogger(__name__)


class UserMemoryFact(BaseModel):
    """A semantic fact or preference learned about a user."""

    fact_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    content: str
    category: Literal["preference", "profile", "topic_interest", "past_query"] = "preference"
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = Field(default_factory=dict)


class LongTermMemoryStore:
    """Stores and retrieves user semantic facts using vector search."""

    def __init__(
        self,
        qdrant_client: Any = None,
        collection_name: str = "muhawir_user_memory",
        embedding_model: Any = None,
    ):
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        # Local in-memory store: user_id -> list[UserMemoryFact]
        self._local_store: dict[str, list[UserMemoryFact]] = {}

    def _embed(self, text: str) -> list[float]:
        """Generate embedding vector for a memory fact or query."""
        if self.embedding_model is not None and hasattr(self.embedding_model, "encode"):
            emb = self.embedding_model.encode(text)
            if hasattr(emb, "tolist"):
                return emb.tolist()
            return list(emb)
        return simple_hash_vector(text, dim=1024)

    async def save_fact(
        self,
        user_id: str,
        content: str,
        category: Literal["preference", "profile", "topic_interest", "past_query"] = "preference",
        metadata: dict[str, Any] | None = None,
    ) -> UserMemoryFact:
        """Save a new memory fact for a user."""
        fact = UserMemoryFact(
            user_id=user_id,
            content=content.strip(),
            category=category,
            metadata=metadata or {},
        )

        # Store in Qdrant if client available
        if self.qdrant_client is not None:
            try:
                from qdrant_client.http import models as rest_models
                vector = self._embed(fact.content)
                point = rest_models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload=fact.model_dump(),
                )
                upsert_call = self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=[point],
                )
                if hasattr(upsert_call, "__await__"):
                    await upsert_call
            except Exception as e:
                logger.warning("Failed to save memory to Qdrant, using local store: %s", e)

        # Also store locally
        if user_id not in self._local_store:
            self._local_store[user_id] = []
        self._local_store[user_id].append(fact)
        logger.info("Saved user memory fact for user=%s: %s", user_id, fact.content[:50])
        return fact

    async def retrieve_user_memories(
        self,
        user_id: str,
        query: str,
        top_k: int = 3,
    ) -> list[UserMemoryFact]:
        """Retrieve the most semantically relevant memories for a user given a query."""
        # Try Qdrant search first
        if self.qdrant_client is not None:
            try:
                from qdrant_client.http import models as rest_models
                query_vec = self._embed(query)
                user_filter = rest_models.Filter(
                    must=[
                        rest_models.FieldCondition(
                            key="user_id",
                            match=rest_models.MatchValue(value=user_id),
                        )
                    ]
                )
                search_call = self.qdrant_client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vec,
                    query_filter=user_filter,
                    limit=top_k,
                )
                hits = await search_call if hasattr(search_call, "__await__") else search_call
                results = []
                for hit in hits:
                    if hasattr(hit, "payload") and hit.payload:
                        results.append(UserMemoryFact(**hit.payload))
                if results:
                    return results
            except Exception as e:
                logger.debug("Qdrant memory search skipped: %s", e)

        # Fallback to local store
        user_facts = self._local_store.get(user_id, [])
        if not user_facts:
            return []

        # Simple keyword overlap / token score for local fallback
        query_words = set(query.lower().split())
        scored_facts = []
        for fact in user_facts:
            fact_words = set(fact.content.lower().split())
            overlap = len(query_words.intersection(fact_words))
            scored_facts.append((overlap, fact))

        scored_facts.sort(key=lambda x: x[0], reverse=True)
        return [f for _, f in scored_facts[:top_k]]

    async def extract_and_save_facts(
        self,
        user_id: str,
        user_query: str,
        assistant_response: str,
        client: Any,
        model: str = "Qwen/Qwen2.5-14B-Instruct",
    ) -> list[UserMemoryFact]:
        """Use lightweight LLM call to extract any enduring user preferences/facts from the turn."""
        if not user_id or len(user_query) < 10:
            return []

        prompt = (
            "Analyze this dialogue turn and determine if the user stated any enduring personal preference, "
            "role/profile, or research topic interest worth remembering for future sessions.\n"
            f"User: {user_query}\n"
            f"Assistant: {assistant_response[:200]}\n\n"
            "If yes, extract 1-2 concise facts. Reply ONLY with JSON array of objects:\n"
            '[{"content": "...", "category": "preference|profile|topic_interest"}]\n'
            "If no enduring facts are present, return an empty array: []"
        )

        extracted_facts: list[UserMemoryFact] = []
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=150,
            )
            raw = resp.choices[0].message.content or "[]"
            m = re.search(r"\[.*\]", raw, re.DOTALL)
            items = json.loads(m.group(0)) if m else []
            if isinstance(items, list):
                for item in items:
                    c = item.get("content", "").strip()
                    cat = item.get("category", "preference")
                    if c and len(c) > 4:
                        fact = await self.save_fact(user_id=user_id, content=c, category=cat)
                        extracted_facts.append(fact)
        except Exception as e:
            logger.debug("Fact extraction skipped: %s", e)

        return extracted_facts

    async def clear_user_memories(self, user_id: str) -> bool:
        """Clear all stored memories for a user."""
        if user_id in self._local_store:
            del self._local_store[user_id]
        return True

    async def format_memories_for_prompt(self, user_id: str, query: str) -> str:
        """Format top relevant memories into a string for injection into the prompt."""
        memories = await self.retrieve_user_memories(user_id=user_id, query=query, top_k=3)
        if not memories:
            return ""
        lines = [f"- {m.content} ({m.category})" for m in memories]
        return "User Profile & Preferences:\n" + "\n".join(lines)


# Singleton
_GLOBAL_LONG_TERM_STORE: LongTermMemoryStore | None = None


def get_long_term_memory_store(
    qdrant_client: Any = None,
    embedding_model: Any = None,
) -> LongTermMemoryStore:
    """Get or create singleton long-term memory store."""
    global _GLOBAL_LONG_TERM_STORE
    if _GLOBAL_LONG_TERM_STORE is None:
        _GLOBAL_LONG_TERM_STORE = LongTermMemoryStore(
            qdrant_client=qdrant_client,
            embedding_model=embedding_model,
        )
    return _GLOBAL_LONG_TERM_STORE


def set_long_term_memory_store(store: LongTermMemoryStore | None) -> None:
    """Set global long-term memory store instance (useful for testing)."""
    global _GLOBAL_LONG_TERM_STORE
    _GLOBAL_LONG_TERM_STORE = store
