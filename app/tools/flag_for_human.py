"""
flag_for_human tool — escalates a conversation when confidence is too low.

Real callable function that persists a flag row via the memory repository.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from app.memory.base import AbstractMemoryRepository, StoredFlag

logger = logging.getLogger(__name__)

_MAX_SNIPPET_LEN = 500
_MAX_REASON_LEN = 2000


@dataclass(frozen=True)
class FlagResult:
    """Structured result after persisting a human-review flag."""

    flag_id: int
    user_id: str
    session_id: str
    reason: str
    message_snippet: str
    confidence_score: float
    resolved: bool
    persisted: bool

    def to_context_string(self) -> str:
        return (
            f"Conversation flagged for human review (flag_id={self.flag_id}). "
            f"Reason: {self.reason}. "
            f"A support representative will follow up with the user."
        )


def flag_for_human(
    user_id: str,
    session_id: str,
    reason: str,
    message_snippet: str,
    confidence_score: float,
    repo: AbstractMemoryRepository,
) -> FlagResult:
    """
    Persist a human-review flag and return structured confirmation.

    Called automatically by the agent when eval confidence is below threshold,
    or when the agent detects an ambiguous / high-risk response.
    """
    if not user_id.strip():
        raise ValueError("user_id is required")
    if not session_id.strip():
        raise ValueError("session_id is required")
    if not reason.strip():
        raise ValueError("reason is required")

    clean_reason = reason.strip()[:_MAX_REASON_LEN]
    clean_snippet = message_snippet.strip()[:_MAX_SNIPPET_LEN]
    bounded_confidence = max(0.0, min(float(confidence_score), 1.0))

    logger.warning(
        "flag_for_human | user=%s session=%s confidence=%.2f reason=%s",
        user_id,
        session_id,
        bounded_confidence,
        clean_reason[:120],
    )

    stored: StoredFlag = repo.save_flag(
        user_id=user_id,
        session_id=session_id,
        reason=clean_reason,
        message_snippet=clean_snippet,
        confidence_score=bounded_confidence,
    )

    result = FlagResult(
        flag_id=stored.id,
        user_id=stored.user_id,
        session_id=stored.session_id,
        reason=stored.reason,
        message_snippet=stored.message_snippet,
        confidence_score=stored.confidence_score,
        resolved=stored.resolved,
        persisted=True,
    )

    logger.info(
        "flag_for_human persisted | flag_id=%s user=%s session=%s",
        result.flag_id,
        user_id,
        session_id,
    )
    return result
