"""
SalesAgent — central orchestration for the persistent sales assistant.

Agent flow (per request):
  1. Retrieve user memory        — get_user_memory tool
  2. Search catalog              — search_catalog tool
  3. Build context               — system prompt with tool outputs
  4. Call LLM API                — Groq / OpenAI / Anthropic via LLMService
  5. Generate response           — assistant text from LLM
  6. Self-evaluate response      — EvalService structured JSON scoring
  7. Save memory                 — persist user + assistant turns
  8. Save eval                   — persist eval scores
  9. Flag low-confidence         — flag_for_human tool when flagged
 10. Return structured response   — ChatResponse with request_id + session_id
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime

from app.config import get_settings
from app.memory.base import AbstractMemoryRepository
from app.models.schemas import ChatResponse, EvalBlock
from app.services.eval_service import EvalRequest, EvalService
from app.services.llm import LLMError, LLMService, LLMTimeoutError, get_llm_service
from app.services.memory_service import MemoryService
from app.tools.flag_for_human import flag_for_human
from app.tools.get_user_memory import get_user_memory
from app.tools.search_catalog import search_catalog

logger = logging.getLogger(__name__)
settings = get_settings()

_SYSTEM_PROMPT = """\
You are a knowledgeable and friendly sales assistant for NexusHQ, a B2B SaaS platform.
Your role is to help prospects understand our pricing, features, and how NexusHQ fits
their needs.

Guidelines:
• Answer ONLY from the catalog context and memory context provided to you.
• If you genuinely don't know something, say so honestly and offer to connect the
  prospect with the sales team at sales@nexushq.com.
• Never invent features, prices, or promises not in the catalog.
• Be conversational but precise.  Use numbers when they help.
• When recommending a plan, explain *why* it fits based on what the user said.
• Always end with a clear next step (book a demo, start a free trial, or contact sales).

CATALOG CONTEXT:
{catalog_context}

MEMORY CONTEXT (prior conversations with this user):
{memory_context}
"""

_MEMORY_UNAVAILABLE = (
    "Memory retrieval failed for this request. Proceed using catalog context only."
)
_CATALOG_UNAVAILABLE = (
    "Catalog search failed for this request. Answer conservatively from general "
    "NexusHQ plan names only (Starter $49/mo, Growth $199/mo, Enterprise $499/mo)."
)


class AgentError(Exception):
    """Raised when the agent cannot complete an LLM generation step."""


class SalesAgent:
    def __init__(
        self,
        repo: AbstractMemoryRepository,
        llm: LLMService | None = None,
        eval_service: EvalService | None = None,
        memory_service: MemoryService | None = None,
    ) -> None:
        self._repo = repo
        self._llm = llm or get_llm_service()
        self._memory = memory_service or MemoryService(repo, llm=self._llm)
        self._eval = eval_service or EvalService(llm=self._llm)

    def chat(
        self,
        user_id: str,
        message: str,
        session_id: str | None = None,
        request_id: str | None = None,
    ) -> ChatResponse:
        """
        Process a single user message and return a grounded, self-evaluated response.

        Each call receives a unique request_id (for tracing) and session_id
        (for conversation grouping across turns).
        """
        request_id = request_id or str(uuid.uuid4())
        session_id = session_id or str(uuid.uuid4())
        tools_called: list[str] = []
        message = message.strip()

        logger.info(
            "[%s] Agent.chat start | user=%s session=%s msg_len=%d",
            request_id,
            user_id,
            session_id,
            len(message),
        )

        # Ensure session exists before tool calls that may write data later.
        self._memory.ensure_session(user_id, session_id, request_id=request_id)

        # ── 1. Retrieve user memory ───────────────────────────────────────────
        memory_context = self._retrieve_memory(user_id, request_id, tools_called)

        # ── 2. Search catalog ─────────────────────────────────────────────────
        catalog_context = self._search_catalog(message, request_id, tools_called)

        # ── 3. Build context ──────────────────────────────────────────────────
        system_prompt = _SYSTEM_PROMPT.format(
            catalog_context=catalog_context,
            memory_context=memory_context,
        )
        logger.debug("[%s] Context built | system_len=%d", request_id, len(system_prompt))

        # ── 4–5. Call LLM API and generate response ─────────────────────────
        assistant_text = self._generate_response(
            request_id=request_id,
            user_id=user_id,
            session_id=session_id,
            system_prompt=system_prompt,
            message=message,
        )

        # ── 6. Self-evaluate response ─────────────────────────────────────────
        eval_block = self._evaluate_response(
            request_id=request_id,
            user_id=user_id,
            session_id=session_id,
            message=message,
            assistant_text=assistant_text,
            catalog_context=catalog_context,
            memory_context=memory_context,
        )

        # ── 7. Save memory ────────────────────────────────────────────────────
        assistant_msg = self._memory.persist_turn(
            user_id,
            session_id,
            message,
            assistant_text,
            request_id=request_id,
        )

        # ── 8. Save eval ──────────────────────────────────────────────────────
        self._memory.save_eval_record(
            user_id=user_id,
            session_id=session_id,
            request_id=request_id,
            groundedness=eval_block.groundedness,
            relevance=eval_block.relevance,
            confidence=eval_block.confidence,
            flagged=eval_block.flagged,
            reasoning=eval_block.reasoning,
            tools_called=tools_called,
            message_id=assistant_msg.id,
        )

        # ── 9. Flag low-confidence responses ──────────────────────────────────
        if eval_block.flagged:
            self._flag_for_review(
                request_id=request_id,
                user_id=user_id,
                session_id=session_id,
                eval_block=eval_block,
                assistant_text=assistant_text,
                tools_called=tools_called,
            )

        # Optional: compress older messages when threshold exceeded.
        self._memory.maybe_compress(user_id, request_id=request_id)

        # ── 10. Return structured response ────────────────────────────────────
        response = ChatResponse(
            response=assistant_text,
            eval=eval_block,
            tools_called=tools_called,
            session_id=session_id,
            user_id=user_id,
            request_id=request_id,
            timestamp=datetime.utcnow(),
        )

        logger.info(
            "[%s] Agent.chat complete | user=%s session=%s tools=%s flagged=%s",
            request_id,
            user_id,
            session_id,
            tools_called,
            eval_block.flagged,
        )
        return response

    # ── Private step helpers ──────────────────────────────────────────────────

    def _retrieve_memory(
        self,
        user_id: str,
        request_id: str,
        tools_called: list[str],
    ) -> str:
        logger.info("[%s] Step 1: retrieve user memory | user=%s", request_id, user_id)
        try:
            context = get_user_memory(user_id, self._repo)
            tools_called.append("get_user_memory")
            return context
        except Exception as exc:
            logger.error(
                "[%s] get_user_memory failed | user=%s error=%s",
                request_id,
                user_id,
                exc,
            )
            tools_called.append("get_user_memory")
            return _MEMORY_UNAVAILABLE

    def _search_catalog(
        self,
        message: str,
        request_id: str,
        tools_called: list[str],
    ) -> str:
        logger.info("[%s] Step 2: search catalog", request_id)
        try:
            context = search_catalog(message)
            tools_called.append("search_catalog")
            return context
        except Exception as exc:
            logger.error("[%s] search_catalog failed | error=%s", request_id, exc)
            tools_called.append("search_catalog")
            return _CATALOG_UNAVAILABLE

    def _generate_response(
        self,
        *,
        request_id: str,
        user_id: str,
        session_id: str,
        system_prompt: str,
        message: str,
    ) -> str:
        logger.info(
            "[%s] Step 4–5: LLM generate | user=%s session=%s provider=%s model=%s",
            request_id,
            user_id,
            session_id,
            self._llm.provider_name,
            self._llm.model_name,
        )
        try:
            assistant_text = self._llm.complete(
                system=system_prompt,
                user_message=message,
                max_tokens=settings.AGENT_MAX_TOKENS,
                temperature=0.3,
            )
        except LLMTimeoutError as exc:
            logger.error(
                "[%s] LLM timeout | user=%s session=%s error=%s",
                request_id,
                user_id,
                session_id,
                exc,
            )
            raise AgentError(f"LLM request timed out: {exc}") from exc
        except LLMError as exc:
            logger.error(
                "[%s] LLM error | user=%s session=%s error=%s",
                request_id,
                user_id,
                session_id,
                exc,
            )
            raise AgentError(f"LLM request failed: {exc}") from exc

        if not assistant_text.strip():
            logger.error("[%s] LLM returned empty response", request_id)
            raise AgentError("LLM returned an empty response")

        logger.info(
            "[%s] Response generated | user=%s session=%s len=%d",
            request_id,
            user_id,
            session_id,
            len(assistant_text),
        )
        return assistant_text.strip()

    def _evaluate_response(
        self,
        *,
        request_id: str,
        user_id: str,
        session_id: str,
        message: str,
        assistant_text: str,
        catalog_context: str,
        memory_context: str,
    ) -> EvalBlock:
        logger.info(
            "[%s] Step 6: self-evaluate | user=%s session=%s",
            request_id,
            user_id,
            session_id,
        )
        return self._eval.evaluate(
            EvalRequest(
                user_message=message,
                assistant_response=assistant_text,
                catalog_context=catalog_context,
                memory_context=memory_context,
                user_id=user_id,
                session_id=session_id,
                request_id=request_id,
            )
        )

    def _flag_for_review(
        self,
        *,
        request_id: str,
        user_id: str,
        session_id: str,
        eval_block: EvalBlock,
        assistant_text: str,
        tools_called: list[str],
    ) -> None:
        logger.warning(
            "[%s] Step 9: flag for human review | user=%s session=%s confidence=%.2f",
            request_id,
            user_id,
            session_id,
            eval_block.confidence,
        )
        try:
            flag_for_human(
                user_id=user_id,
                session_id=session_id,
                reason=(
                    f"Low confidence response. Scores: G={eval_block.groundedness:.2f} "
                    f"R={eval_block.relevance:.2f} C={eval_block.confidence:.2f}. "
                    f"Reason: {eval_block.reasoning}"
                ),
                message_snippet=assistant_text[:300],
                confidence_score=eval_block.confidence,
                repo=self._repo,
            )
            tools_called.append("flag_for_human")
        except Exception as exc:
            logger.error(
                "[%s] flag_for_human failed | user=%s session=%s error=%s",
                request_id,
                user_id,
                session_id,
                exc,
            )
