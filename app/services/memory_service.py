"""
MemoryService — session management, turn persistence, and optional compression.

Keeps the context window bounded by summarising older messages when the
message count exceeds MEMORY_SUMMARY_THRESHOLD.
"""
from __future__ import annotations

import logging

from app.config import get_settings
from app.memory.base import AbstractMemoryRepository, StoredMessage, StoredSession
from app.services.llm import LLMError, LLMService, get_llm_service

logger = logging.getLogger(__name__)
settings = get_settings()

_SUMMARY_SYSTEM = """\
You are a memory compression assistant for a sales chatbot.
Given a conversation history, write a concise 3-5 sentence factual summary
covering: what plans the user asked about, any preferences they expressed,
concerns raised, and any commitments made.  Be specific (include plan names,
prices, features mentioned).  Do NOT include pleasantries or filler.
"""


class MemoryService:
    """High-level memory operations used by the sales agent."""

    def __init__(
        self,
        repo: AbstractMemoryRepository,
        llm: LLMService | None = None,
    ) -> None:
        self._repo = repo
        self._llm = llm or get_llm_service()

    def ensure_session(
        self,
        user_id: str,
        session_id: str,
        *,
        request_id: str,
    ) -> StoredSession:
        """Create or validate a conversation session."""
        logger.info(
            "[%s] MemoryService.ensure_session | user=%s session=%s",
            request_id,
            user_id,
            session_id,
        )
        try:
            session = self._repo.create_session(user_id, session_id)
            logger.debug(
                "[%s] Session ready | user=%s session=%s",
                request_id,
                user_id,
                session.id,
            )
            return session
        except Exception as exc:
            logger.error(
                "[%s] Failed to ensure session | user=%s session=%s error=%s",
                request_id,
                user_id,
                session_id,
                exc,
            )
            raise

    def persist_turn(
        self,
        user_id: str,
        session_id: str,
        user_message: str,
        assistant_message: str,
        *,
        request_id: str,
    ) -> StoredMessage:
        """
        Persist the user message and assistant response for a single turn.

        Returns the stored assistant message (used to link eval records).
        """
        logger.info(
            "[%s] MemoryService.persist_turn | user=%s session=%s",
            request_id,
            user_id,
            session_id,
        )
        try:
            self._repo.save_message(user_id, session_id, "user", user_message)
            assistant_msg = self._repo.save_message(
                user_id, session_id, "assistant", assistant_message
            )
            logger.info(
                "[%s] Turn persisted | user=%s session=%s assistant_msg_id=%s",
                request_id,
                user_id,
                session_id,
                assistant_msg.id,
            )
            return assistant_msg
        except Exception as exc:
            logger.error(
                "[%s] Failed to persist turn | user=%s session=%s error=%s",
                request_id,
                user_id,
                session_id,
                exc,
            )
            raise

    def save_eval_record(
        self,
        *,
        user_id: str,
        session_id: str,
        request_id: str,
        groundedness: float,
        relevance: float,
        confidence: float,
        flagged: bool,
        reasoning: str,
        tools_called: list[str],
        message_id: int | None,
    ) -> None:
        """Persist self-evaluation scores linked to the assistant message."""
        logger.info(
            "[%s] MemoryService.save_eval_record | user=%s session=%s flagged=%s",
            request_id,
            user_id,
            session_id,
            flagged,
        )
        try:
            self._repo.save_eval(
                user_id=user_id,
                session_id=session_id,
                groundedness=groundedness,
                relevance=relevance,
                confidence=confidence,
                flagged=flagged,
                reasoning=reasoning,
                tools_called=tools_called,
                message_id=message_id,
            )
        except Exception as exc:
            logger.error(
                "[%s] Failed to save eval | user=%s session=%s error=%s",
                request_id,
                user_id,
                session_id,
                exc,
            )
            raise

    def maybe_compress(self, user_id: str, *, request_id: str) -> bool:
        """
        If the user has more than MEMORY_SUMMARY_THRESHOLD messages,
        compress the oldest half into a summary.  Returns True if compression ran.
        """
        messages = self._repo.get_messages(user_id)
        threshold = settings.MEMORY_SUMMARY_THRESHOLD

        if len(messages) < threshold:
            logger.debug(
                "[%s] Compression skipped | user=%s count=%d threshold=%d",
                request_id,
                user_id,
                len(messages),
                threshold,
            )
            return False

        cutoff = len(messages) // 2
        to_compress = messages[:cutoff]

        history_text = "\n".join(
            f"{'User' if m.role == 'user' else 'Assistant'}: {m.content}"
            for m in to_compress
        )

        logger.info(
            "[%s] Compressing %d messages | user=%s",
            request_id,
            len(to_compress),
            user_id,
        )

        try:
            summary = self._llm.complete(
                system=_SUMMARY_SYSTEM,
                user_message=history_text,
                max_tokens=400,
                temperature=0.2,
            )
            self._repo.save_summary(user_id, summary, len(to_compress))
            logger.info(
                "[%s] Compression complete | user=%s messages=%d",
                request_id,
                user_id,
                len(to_compress),
            )
            return True
        except LLMError as exc:
            logger.error(
                "[%s] Compression LLM error | user=%s error=%s",
                request_id,
                user_id,
                exc,
            )
            return False
        except Exception as exc:
            logger.error(
                "[%s] Compression failed | user=%s error=%s",
                request_id,
                user_id,
                exc,
            )
            return False


# Backward-compatible alias used by earlier phases.
MemoryCompressionService = MemoryService
