"""
get_user_memory tool — retrieves persisted user context from the memory repository.

Real callable function used by the agent loop (not prompt simulation).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.config import get_settings
from app.memory.base import AbstractMemoryRepository, StoredMessage

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass(frozen=True)
class MemoryContext:
    """Structured memory payload for a user."""

    user_id: str
    has_history: bool
    summary: str | None
    recent_messages: list[StoredMessage]
    session_ids: list[str]
    message_count: int
    truncated_to: int

    def to_context_string(self) -> str:
        """Format memory for LLM / agent context injection."""
        if not self.has_history:
            return "No prior conversation history for this user."

        parts: list[str] = [
            f"=== User Memory Context (user_id={self.user_id}) ===",
            f"Sessions: {len(self.session_ids)} | "
            f"Recent messages shown: {self.message_count} (limit {self.truncated_to})",
        ]

        if self.session_ids:
            parts.append(f"Session IDs (most recent first): {', '.join(self.session_ids)}")

        if self.summary:
            parts.append(f"[Earlier conversation summary]\n{self.summary}")

        if self.recent_messages:
            history_lines: list[str] = []
            for msg in self.recent_messages:
                label = "User" if msg.role == "user" else "Assistant"
                ts = msg.timestamp.strftime("%Y-%m-%d %H:%M")
                session_hint = f" session={msg.session_id[:8]}…" if msg.session_id else ""
                history_lines.append(f"[{ts}{session_hint}] {label}: {msg.content}")
            parts.append("[Recent conversation history]\n" + "\n".join(history_lines))

        return "\n\n".join(parts)


def get_user_memory_structured(
    user_id: str,
    repo: AbstractMemoryRepository,
    *,
    limit: int | None = None,
) -> MemoryContext:
    """
    Query the memory repository and return structured context for *user_id*.
    """
    max_messages = limit if limit is not None else settings.MEMORY_MAX_MESSAGES
    logger.info("get_user_memory_structured | user_id=%s limit=%d", user_id, max_messages)

    summary = repo.get_summary(user_id)
    recent_messages = repo.get_recent_messages(user_id, max_messages)
    session_ids = repo.get_sessions(user_id)

    has_history = bool(summary or recent_messages)

    context = MemoryContext(
        user_id=user_id,
        has_history=has_history,
        summary=summary,
        recent_messages=recent_messages,
        session_ids=session_ids,
        message_count=len(recent_messages),
        truncated_to=max_messages,
    )

    logger.info(
        "get_user_memory_structured | user=%s messages=%d sessions=%d summary=%s",
        user_id,
        len(recent_messages),
        len(session_ids),
        bool(summary),
    )
    return context


def get_user_memory(user_id: str, repo: AbstractMemoryRepository) -> str:
    """
    Fetch recent messages and any compressed summary for *user_id*.

    Returns a formatted context string the agent injects into its system prompt.
    """
    logger.info("get_user_memory called | user_id=%s", user_id)
    context = get_user_memory_structured(user_id, repo)
    return context.to_context_string()
