"""
SQLAlchemy ORM models for the Sales Assistant Agent.

Tables:
  sessions          — one row per conversation session (UUID primary key)
  messages          — every user/assistant turn, FK → sessions
  eval_results      — self-eval scores per assistant response
  flags             — low-confidence escalations for human review
  memory_summaries  — compressed rolling context per user
"""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

if TYPE_CHECKING:
    pass


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


class ChatSession(Base):
    """A single conversation session for a user (UUID primary key)."""

    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    ended_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=False),
        nullable=True,
    )

    messages: Mapped[list[Message]] = relationship(
        "Message",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="Message.timestamp",
    )
    eval_results: Mapped[list[EvalResult]] = relationship(
        "EvalResult",
        back_populates="session",
        cascade="all, delete-orphan",
    )
    flags: Mapped[list[Flag]] = relationship(
        "Flag",
        back_populates="session",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("ix_sessions_user_id_created_at", "user_id", "created_at"),
    )


class Message(Base):
    """Stores every chat turn for every user across all sessions."""

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    session_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)  # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=False),
        server_default=func.now(),
        nullable=False,
    )

    session: Mapped[ChatSession] = relationship("ChatSession", back_populates="messages")
    eval_result: Mapped[EvalResult | None] = relationship(
        "EvalResult",
        back_populates="message",
        uselist=False,
    )

    __table_args__ = (
        Index("ix_messages_user_id_timestamp", "user_id", "timestamp"),
        Index("ix_messages_session_id_timestamp", "session_id", "timestamp"),
    )


class EvalResult(Base):
    """Self-evaluation scores for every assistant response."""

    __tablename__ = "eval_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    session_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    message_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("messages.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    groundedness: Mapped[float] = mapped_column(Float, nullable=False)
    relevance: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    flagged: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    reasoning: Mapped[str] = mapped_column(Text, nullable=False)
    tools_called: Mapped[str] = mapped_column(Text, nullable=False)  # JSON-serialised list
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False),
        server_default=func.now(),
        nullable=False,
    )

    session: Mapped[ChatSession] = relationship("ChatSession", back_populates="eval_results")
    message: Mapped[Message | None] = relationship("Message", back_populates="eval_result")

    __table_args__ = (
        Index("ix_eval_results_user_id_created_at", "user_id", "created_at"),
        Index("ix_eval_results_session_id_created_at", "session_id", "created_at"),
    )


class Flag(Base):
    """Human-review flags raised when confidence drops below threshold."""

    __tablename__ = "flags"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    session_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    message_snippet: Mapped[str] = mapped_column(Text, nullable=False)
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False),
        server_default=func.now(),
        nullable=False,
    )
    resolved: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    session: Mapped[ChatSession] = relationship("ChatSession", back_populates="flags")

    __table_args__ = (
        Index("ix_flags_user_id_created_at", "user_id", "created_at"),
        Index("ix_flags_resolved_created_at", "resolved", "created_at"),
    )


class MemorySummary(Base):
    """Compressed summaries of older conversation context (bonus feature)."""

    __tablename__ = "memory_summaries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    message_count_compressed: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
