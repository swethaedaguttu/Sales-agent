"""
SQLAlchemy-backed memory repository.

This is the *only* memory file that knows about SQLAlchemy.  To swap in Postgres,
Redis, or Mem0, create a new file (e.g. memory/postgres_repository.py or
memory/mem0_repository.py) that implements AbstractMemoryRepository and update
memory/factory.py.  Zero business-logic changes required.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime

from sqlalchemy.orm import Session

from app.db.orm_models import ChatSession, EvalResult, Flag, MemorySummary, Message
from app.memory.base import (
    AbstractMemoryRepository,
    StoredEval,
    StoredFlag,
    StoredMessage,
    StoredSession,
)

logger = logging.getLogger(__name__)


class SQLiteMemoryRepository(AbstractMemoryRepository):
    """SQLite / Postgres-compatible SQLAlchemy implementation."""

    def __init__(self, db: Session) -> None:
        self._db = db

    # ── Sessions ──────────────────────────────────────────────────────────────

    def create_session(self, user_id: str, session_id: str) -> StoredSession:
        existing = self._db.get(ChatSession, session_id)
        if existing is not None:
            if existing.user_id != user_id:
                raise ValueError(
                    f"Session {session_id!r} belongs to user {existing.user_id!r}, "
                    f"not {user_id!r}"
                )
            return self._to_stored_session(existing)

        session = ChatSession(id=session_id, user_id=user_id)
        self._db.add(session)
        self._db.commit()
        self._db.refresh(session)
        logger.debug("Created session id=%s user=%s", session_id, user_id)
        return self._to_stored_session(session)

    def get_sessions(self, user_id: str) -> list[str]:
        rows = (
            self._db.query(ChatSession.id)
            .filter(ChatSession.user_id == user_id)
            .order_by(ChatSession.updated_at.desc())
            .all()
        )
        return [row[0] for row in rows]

    # ── Messages ──────────────────────────────────────────────────────────────

    def save_message(
        self,
        user_id: str,
        session_id: str,
        role: str,
        content: str,
    ) -> StoredMessage:
        session = self._ensure_session(user_id, session_id)
        session.updated_at = datetime.utcnow()

        msg = Message(
            user_id=user_id,
            session_id=session_id,
            role=role,
            content=content,
        )
        self._db.add(msg)
        self._db.commit()
        self._db.refresh(msg)
        logger.debug("Saved message id=%s user=%s role=%s", msg.id, user_id, role)
        return self._to_stored_message(msg)

    def get_messages(
        self,
        user_id: str,
        limit: int | None = None,
    ) -> list[StoredMessage]:
        if limit is not None:
            return self.get_recent_messages(user_id, limit)

        rows = (
            self._db.query(Message)
            .filter(Message.user_id == user_id)
            .order_by(Message.timestamp.asc())
            .all()
        )
        return [self._to_stored_message(row) for row in rows]

    def get_recent_messages(self, user_id: str, limit: int) -> list[StoredMessage]:
        if limit <= 0:
            return []

        rows = (
            self._db.query(Message)
            .filter(Message.user_id == user_id)
            .order_by(Message.timestamp.desc())
            .limit(limit)
            .all()
        )
        return [self._to_stored_message(row) for row in reversed(rows)]

    def clear_user_memory(self, user_id: str) -> tuple[int, int]:
        msg_count = (
            self._db.query(Message)
            .filter(Message.user_id == user_id)
            .count()
        )
        session_count = (
            self._db.query(ChatSession)
            .filter(ChatSession.user_id == user_id)
            .count()
        )

        # Explicit deletes — portable across SQLite/Postgres and safe without CASCADE.
        self._db.query(Flag).filter(Flag.user_id == user_id).delete(synchronize_session=False)
        self._db.query(EvalResult).filter(EvalResult.user_id == user_id).delete(
            synchronize_session=False
        )
        self._db.query(Message).filter(Message.user_id == user_id).delete(synchronize_session=False)
        self._db.query(MemorySummary).filter(MemorySummary.user_id == user_id).delete(
            synchronize_session=False
        )
        self._db.query(ChatSession).filter(ChatSession.user_id == user_id).delete(
            synchronize_session=False
        )
        self._db.commit()

        logger.info(
            "Cleared memory for user=%s: %d messages, %d sessions",
            user_id,
            msg_count,
            session_count,
        )
        return msg_count, session_count

    # ── Eval ──────────────────────────────────────────────────────────────────

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
        self._ensure_session(user_id, session_id)

        ev = EvalResult(
            user_id=user_id,
            session_id=session_id,
            message_id=message_id,
            groundedness=groundedness,
            relevance=relevance,
            confidence=confidence,
            flagged=flagged,
            reasoning=reasoning,
            tools_called=json.dumps(tools_called),
        )
        self._db.add(ev)
        self._db.commit()
        self._db.refresh(ev)
        logger.debug(
            "Saved eval id=%s user=%s confidence=%.2f flagged=%s",
            ev.id,
            user_id,
            confidence,
            flagged,
        )
        return self._to_stored_eval(ev)

    def get_evals(self, user_id: str) -> list[StoredEval]:
        rows = (
            self._db.query(EvalResult)
            .filter(EvalResult.user_id == user_id)
            .order_by(EvalResult.created_at.asc())
            .all()
        )
        return [self._to_stored_eval(row) for row in rows]

    # ── Flags ─────────────────────────────────────────────────────────────────

    def save_flag(
        self,
        user_id: str,
        session_id: str,
        reason: str,
        message_snippet: str,
        confidence_score: float,
    ) -> StoredFlag:
        self._ensure_session(user_id, session_id)

        flag = Flag(
            user_id=user_id,
            session_id=session_id,
            reason=reason,
            message_snippet=message_snippet,
            confidence_score=confidence_score,
        )
        self._db.add(flag)
        self._db.commit()
        self._db.refresh(flag)
        logger.warning(
            "FLAG raised user=%s session=%s reason=%s confidence=%.2f",
            user_id,
            session_id,
            reason,
            confidence_score,
        )
        return self._to_stored_flag(flag)

    def get_flags(
        self,
        resolved: bool | None = None,
        user_id: str | None = None,
    ) -> list[StoredFlag]:
        q = self._db.query(Flag)
        if resolved is not None:
            q = q.filter(Flag.resolved == resolved)
        if user_id is not None:
            q = q.filter(Flag.user_id == user_id)
        rows = q.order_by(Flag.created_at.desc()).all()
        return [self._to_stored_flag(row) for row in rows]

    # ── Memory summary ────────────────────────────────────────────────────────

    def get_summary(self, user_id: str) -> str | None:
        row = (
            self._db.query(MemorySummary)
            .filter(MemorySummary.user_id == user_id)
            .order_by(MemorySummary.updated_at.desc())
            .first()
        )
        return row.summary if row else None

    def save_summary(self, user_id: str, summary: str, message_count: int) -> None:
        existing = (
            self._db.query(MemorySummary)
            .filter(MemorySummary.user_id == user_id)
            .first()
        )
        if existing:
            existing.summary = summary
            existing.message_count_compressed = message_count
            existing.updated_at = datetime.utcnow()
        else:
            self._db.add(
                MemorySummary(
                    user_id=user_id,
                    summary=summary,
                    message_count_compressed=message_count,
                )
            )
        self._db.commit()

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _ensure_session(self, user_id: str, session_id: str) -> ChatSession:
        session = self._db.get(ChatSession, session_id)
        if session is None:
            session = ChatSession(id=session_id, user_id=user_id)
            self._db.add(session)
            self._db.flush()
            logger.debug("Auto-created session id=%s user=%s", session_id, user_id)
            return session

        if session.user_id != user_id:
            raise ValueError(
                f"Session {session_id!r} belongs to user {session.user_id!r}, "
                f"not {user_id!r}"
            )
        return session

    @staticmethod
    def _to_stored_session(session: ChatSession) -> StoredSession:
        return StoredSession(
            id=session.id,
            user_id=session.user_id,
            created_at=session.created_at,
            updated_at=session.updated_at,
            ended_at=session.ended_at,
        )

    @staticmethod
    def _to_stored_message(message: Message) -> StoredMessage:
        return StoredMessage(
            id=message.id,
            user_id=message.user_id,
            session_id=message.session_id,
            role=message.role,
            content=message.content,
            timestamp=message.timestamp,
        )

    @staticmethod
    def _to_stored_eval(eval_result: EvalResult) -> StoredEval:
        return StoredEval(
            id=eval_result.id,
            user_id=eval_result.user_id,
            session_id=eval_result.session_id,
            message_id=eval_result.message_id,
            groundedness=eval_result.groundedness,
            relevance=eval_result.relevance,
            confidence=eval_result.confidence,
            flagged=eval_result.flagged,
            reasoning=eval_result.reasoning,
            tools_called=json.loads(eval_result.tools_called),
            created_at=eval_result.created_at,
        )

    @staticmethod
    def _to_stored_flag(flag: Flag) -> StoredFlag:
        return StoredFlag(
            id=flag.id,
            user_id=flag.user_id,
            session_id=flag.session_id,
            reason=flag.reason,
            message_snippet=flag.message_snippet,
            confidence_score=flag.confidence_score,
            created_at=flag.created_at,
            resolved=flag.resolved,
        )
