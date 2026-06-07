from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ── Catalog ──────────────────────────────────────────────────────────────────

class CatalogResponse(BaseModel):
    catalog: dict[str, Any]


# ── Eval ─────────────────────────────────────────────────────────────────────

class EvalBlock(BaseModel):
    groundedness: float = Field(..., ge=0.0, le=1.0)
    relevance: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    flagged: bool
    reasoning: str


class EvalResultRecord(BaseModel):
    """Persisted eval row returned by aggregation endpoints."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: str
    session_id: str
    message_id: int | None = None
    groundedness: float
    relevance: float
    confidence: float
    flagged: bool
    reasoning: str
    tools_called: list[str]
    created_at: datetime


class EvalSummary(BaseModel):
    total_responses: int
    flagged_count: int
    avg_groundedness: float
    avg_relevance: float
    avg_confidence: float
    high_confidence_pct: float  # % responses with confidence >= 0.80


# ── Sessions ─────────────────────────────────────────────────────────────────

class SessionRecord(BaseModel):
    """A conversation session (UUID id)."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    user_id: str
    created_at: datetime
    updated_at: datetime
    ended_at: datetime | None = None

    @field_validator("id", mode="before")
    @classmethod
    def coerce_uuid(cls, value: str | UUID) -> str:
        return str(value)


# ── Chat ──────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: str | None = Field(
        default=None,
        description="Optional: resume a specific session. Auto-generated UUID if omitted.",
    )

    @field_validator("session_id", mode="before")
    @classmethod
    def normalize_session_id(cls, value: str | UUID | None) -> str | None:
        if value is None:
            return None
        return str(value)


class ChatResponse(BaseModel):
    response: str
    eval: EvalBlock
    tools_called: list[str]
    session_id: str
    user_id: str
    request_id: str
    timestamp: datetime


# ── History ───────────────────────────────────────────────────────────────────

class MessageRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    session_id: str
    role: str  # "user" | "assistant"
    content: str
    timestamp: datetime

    @field_validator("session_id", mode="before")
    @classmethod
    def coerce_session_id(cls, value: str | UUID) -> str:
        return str(value)


class HistoryResponse(BaseModel):
    user_id: str
    total_messages: int
    sessions: list[str]
    messages: list[MessageRecord]


# ── Flags ─────────────────────────────────────────────────────────────────────

class FlagRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: str
    session_id: str
    reason: str
    message_snippet: str
    confidence_score: float
    created_at: datetime
    resolved: bool

    @field_validator("session_id", mode="before")
    @classmethod
    def coerce_session_id(cls, value: str | UUID) -> str:
        return str(value)


class FlagsResponse(BaseModel):
    total: int
    unresolved: int
    flags: list[FlagRecord]


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    db: str
    model: str
    uptime_seconds: float


# ── Memory ────────────────────────────────────────────────────────────────────

class MemoryDeleteResponse(BaseModel):
    user_id: str
    deleted_messages: int
    deleted_sessions: int
    status: str
