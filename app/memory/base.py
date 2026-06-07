"""
Abstract base class for the memory layer.

Swapping backends (SQLite → Postgres, Mem0) means implementing this
interface and updating the DI binding in app/memory/factory.py — nothing else
changes in the business logic.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class StoredMessage:
    id: int
    user_id: str
    session_id: str
    role: str
    content: str
    timestamp: datetime


@dataclass(frozen=True)
class StoredSession:
    id: str
    user_id: str
    created_at: datetime
    updated_at: datetime
    ended_at: datetime | None = None


@dataclass(frozen=True)
class StoredEval:
    id: int
    user_id: str
    session_id: str
    groundedness: float
    relevance: float
    confidence: float
    flagged: bool
    reasoning: str
    tools_called: list[str]
    created_at: datetime
    message_id: int | None = None


@dataclass(frozen=True)
class StoredFlag:
    id: int
    user_id: str
    session_id: str
    reason: str
    message_snippet: str
    confidence_score: float
    created_at: datetime
    resolved: bool


class AbstractMemoryRepository(ABC):
    """
    Interface every memory backend must satisfy.

    Implementations must not leak storage details (SQL, HTTP, etc.) to callers.
  """

    # ── Sessions ──────────────────────────────────────────────────────────────

    @abstractmethod
    def create_session(self, user_id: str, session_id: str) -> StoredSession:
        """Create (or return existing) session for *user_id*."""
        ...

    @abstractmethod
    def get_sessions(self, user_id: str) -> list[str]:
        """Return session IDs for *user_id*, most-recent activity first."""
        ...

    # ── Messages ──────────────────────────────────────────────────────────────

    @abstractmethod
    def save_message(
        self,
        user_id: str,
        session_id: str,
        role: str,
        content: str,
    ) -> StoredMessage:
        ...

    @abstractmethod
    def get_messages(
        self,
        user_id: str,
        limit: int | None = None,
    ) -> list[StoredMessage]:
        """Return messages for *user_id*, oldest-first. Optional tail limit."""
        ...

    @abstractmethod
    def get_recent_messages(self, user_id: str, limit: int) -> list[StoredMessage]:
        """Return the *limit* most recent messages, oldest-first."""
        ...

    @abstractmethod
    def clear_user_memory(self, user_id: str) -> tuple[int, int]:
        """
        GDPR-style wipe of all persisted data for *user_id*.

        Returns (deleted_message_count, deleted_session_count).
        """
        ...

    # ── Eval ──────────────────────────────────────────────────────────────────

    @abstractmethod
    def save_eval(
        self,
        user_id: str,
        session_id: str,
        groundedness: float,
        relevance: float,
        confidence: float,
        flagged: bool,
        reasoning: str,
        tools_called: list[str],
        message_id: int | None = None,
    ) -> StoredEval:
        ...

    @abstractmethod
    def get_evals(self, user_id: str) -> list[StoredEval]:
        ...

    # ── Flags ─────────────────────────────────────────────────────────────────

    @abstractmethod
    def save_flag(
        self,
        user_id: str,
        session_id: str,
        reason: str,
        message_snippet: str,
        confidence_score: float,
    ) -> StoredFlag:
        ...

    @abstractmethod
    def get_flags(
        self,
        resolved: bool | None = None,
        user_id: str | None = None,
    ) -> list[StoredFlag]:
        ...

    # ── Memory summary ────────────────────────────────────────────────────────

    @abstractmethod
    def get_summary(self, user_id: str) -> str | None:
        ...

    @abstractmethod
    def save_summary(self, user_id: str, summary: str, message_count: int) -> None:
        ...
