"""Short-term conversation memory backed by Redis with in-memory fallback.

Provides a sliding-window message history for session-based multi-turn conversations,
supporting both English and Arabic dialogues.
"""

from datetime import datetime, timezone
import json
import logging
from typing import Any, Literal
from pydantic import BaseModel, Field

from src.config import get_settings

logger = logging.getLogger(__name__)


class ConversationMessage(BaseModel):
    """Single message in a conversational session."""

    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationSession(BaseModel):
    """Full session object containing metadata and messages."""

    session_id: str
    messages: list[ConversationMessage] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class RedisConversationMemory:
    """Manages short-term conversation memory using Redis with in-memory fallback."""

    def __init__(self, redis_client: Any = None, default_ttl: int = 86400):
        self.redis_client = redis_client
        self.default_ttl = default_ttl
        # Local fallback store: session_id -> list[dict]
        self._local_store: dict[str, list[dict[str, Any]]] = {}

    def _make_key(self, session_id: str) -> str:
        return f"muhawir:session:{session_id}:messages"

    async def add_message(
        self,
        session_id: str,
        role: Literal["user", "assistant", "system"],
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a single message to the session."""
        msg = ConversationMessage(role=role, content=content, metadata=metadata or {})
        msg_dict = msg.model_dump()

        if self.redis_client is not None:
            key = self._make_key(session_id)
            try:
                msg_json = json.dumps(msg_dict)
                # RPUSH to append to list
                rpush_call = self.redis_client.rpush(key, msg_json)
                if hasattr(rpush_call, "__await__"):
                    await rpush_call

                expire_call = self.redis_client.expire(key, self.default_ttl)
                if hasattr(expire_call, "__await__"):
                    await expire_call
                return
            except Exception as e:
                logger.warning("Redis add_message failed, using local store: %s", e)

        # Fallback to local store
        if session_id not in self._local_store:
            self._local_store[session_id] = []
        self._local_store[session_id].append(msg_dict)

    async def add_turn(
        self,
        session_id: str,
        user_query: str,
        assistant_response: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Helper to add both user and assistant turns in sequence."""
        await self.add_message(session_id, "user", user_query, metadata=metadata)
        await self.add_message(session_id, "assistant", assistant_response, metadata=metadata)

    async def get_history(
        self,
        session_id: str,
        max_messages: int = 10,
    ) -> list[ConversationMessage]:
        """Fetch the most recent messages for a session (sliding window)."""
        raw_messages: list[dict[str, Any]] = []

        if self.redis_client is not None:
            key = self._make_key(session_id)
            try:
                # Get the last max_messages
                lrange_call = self.redis_client.lrange(key, -max_messages, -1)
                items = await lrange_call if hasattr(lrange_call, "__await__") else lrange_call
                for item in items:
                    raw_str = item.decode("utf-8") if isinstance(item, bytes) else str(item)
                    raw_messages.append(json.loads(raw_str))
                return [ConversationMessage(**m) for m in raw_messages]
            except Exception as e:
                logger.warning("Redis get_history failed, reading from local store: %s", e)

        # Fallback to local store
        local_msgs = self._local_store.get(session_id, [])
        slice_start = max(0, len(local_msgs) - max_messages)
        return [ConversationMessage(**m) for m in local_msgs[slice_start:]]

    async def clear_session(self, session_id: str) -> bool:
        """Clear all messages for a given session."""
        success = True
        if self.redis_client is not None:
            key = self._make_key(session_id)
            try:
                del_call = self.redis_client.delete(key)
                if hasattr(del_call, "__await__"):
                    await del_call
            except Exception as e:
                logger.warning("Redis clear_session error: %s", e)
                success = False

        if session_id in self._local_store:
            del self._local_store[session_id]

        return success

    async def format_history_for_prompt(
        self,
        session_id: str,
        max_messages: int = 6,
    ) -> str:
        """Format the recent dialogue into a human-readable prompt string."""
        history = await self.get_history(session_id, max_messages=max_messages)
        if not history:
            return "No previous dialogue."

        lines = []
        for msg in history:
            prefix = "User" if msg.role == "user" else "Assistant"
            lines.append(f"{prefix}: {msg.content}")
        return "\n".join(lines)


# Singleton instance
_GLOBAL_CONVERSATION_MEMORY: RedisConversationMemory | None = None


def get_conversation_memory(redis_client: Any = None) -> RedisConversationMemory:
    """Get or create singleton conversation memory manager."""
    global _GLOBAL_CONVERSATION_MEMORY
    if _GLOBAL_CONVERSATION_MEMORY is None:
        _GLOBAL_CONVERSATION_MEMORY = RedisConversationMemory(redis_client=redis_client)
    elif redis_client is not None and _GLOBAL_CONVERSATION_MEMORY.redis_client is None:
        _GLOBAL_CONVERSATION_MEMORY.redis_client = redis_client
    return _GLOBAL_CONVERSATION_MEMORY


def set_conversation_memory(memory: RedisConversationMemory | None) -> None:
    """Set global conversation memory instance (useful for testing)."""
    global _GLOBAL_CONVERSATION_MEMORY
    _GLOBAL_CONVERSATION_MEMORY = memory
