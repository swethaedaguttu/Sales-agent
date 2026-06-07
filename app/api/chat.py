"""
Chat API routes — conversational endpoints with memory, eval, and GDPR reset.
"""
from __future__ import annotations

import logging
import re
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request, status
from sqlalchemy.orm import Session

from app.agents.sales_agent import AgentError, SalesAgent
from app.db.session import get_db
from app.memory.base import AbstractMemoryRepository
from app.memory.factory import get_memory_repository
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    EvalSummary,
    HistoryResponse,
    MemoryDeleteResponse,
    MessageRecord,
)
from app.services.llm import LLMTimeoutError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

# Path parameter validation — alphanumeric, hyphen, underscore.
_USER_ID_PATTERN = re.compile(r"^[\w\-]+$")

UserIdPath = Annotated[
    str,
    Path(
        min_length=1,
        max_length=255,
        description="Unique identifier for the end user (letters, numbers, `_`, `-`).",
        examples=["user-123", "acme_corp_demo"],
    ),
]


# ── Dependency injection ──────────────────────────────────────────────────────


def get_memory_repo(db: Session = Depends(get_db)) -> AbstractMemoryRepository:
    """Inject the configured memory repository (SQLite / Postgres / Mem0)."""
    return get_memory_repository(db)


def get_sales_agent(repo: AbstractMemoryRepository = Depends(get_memory_repo)) -> SalesAgent:
    """Inject a SalesAgent bound to the request-scoped memory repository."""
    return SalesAgent(repo)


def get_request_id(request: Request) -> str:
    """Resolve correlation ID from header or generate a new one."""
    header_id = request.headers.get("X-Request-ID", "").strip()
    return header_id or str(uuid.uuid4())


# ── POST /chat/{user_id} ──────────────────────────────────────────────────────


@router.post(
    "/{user_id}",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Send a message and receive a grounded response with self-eval scores",
    description=(
        "Runs the full agent pipeline: memory retrieval, catalog search, LLM generation, "
        "self-evaluation, persistence, and optional human-review flagging."
    ),
    responses={
        status.HTTP_200_OK: {"description": "Assistant response with eval block"},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Invalid request body or path params"},
        status.HTTP_502_BAD_GATEWAY: {"description": "LLM provider error"},
        status.HTTP_504_GATEWAY_TIMEOUT: {"description": "LLM request timed out"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Unexpected server error"},
    },
)
async def post_chat(
    user_id: UserIdPath,
    body: ChatRequest,
    request_id: Annotated[str, Depends(get_request_id)],
    agent: SalesAgent = Depends(get_sales_agent),
) -> ChatResponse:
    _validate_user_id(user_id)

    logger.info(
        "[%s] POST /chat/%s | session=%s message_len=%d",
        request_id,
        user_id,
        body.session_id,
        len(body.message),
    )

    try:
        return agent.chat(
            user_id=user_id,
            message=body.message,
            session_id=body.session_id,
            request_id=request_id,
        )
    except AgentError as exc:
        error_msg = str(exc)
        if "timed out" in error_msg.lower():
            logger.error("[%s] Agent timeout | user=%s error=%s", request_id, user_id, exc)
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail={"message": error_msg, "request_id": request_id},
            ) from exc
        logger.error("[%s] Agent error | user=%s error=%s", request_id, user_id, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={"message": error_msg, "request_id": request_id},
        ) from exc
    except LLMTimeoutError as exc:
        logger.error("[%s] LLM timeout | user=%s error=%s", request_id, user_id, exc)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail={"message": str(exc), "request_id": request_id},
        ) from exc
    except Exception as exc:
        logger.exception("[%s] Unexpected agent error | user=%s", request_id, user_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": "An unexpected error occurred.", "request_id": request_id},
        ) from exc


# ── GET /chat/{user_id}/history ───────────────────────────────────────────────


@router.get(
    "/{user_id}/history",
    response_model=HistoryResponse,
    status_code=status.HTTP_200_OK,
    summary="Full conversation history across all sessions for a user",
    description=(
        "Returns all messages for the user across every session, oldest-first. "
        "Use the `limit` query parameter to cap the number of most recent messages."
    ),
    responses={
        status.HTTP_200_OK: {"description": "Conversation history (may be empty)"},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Invalid path or query parameters"},
    },
)
async def get_history(
    user_id: UserIdPath,
    request_id: Annotated[str, Depends(get_request_id)],
    repo: AbstractMemoryRepository = Depends(get_memory_repo),
    limit: Annotated[
        int,
        Query(ge=1, le=1000, description="Maximum number of most recent messages to return"),
    ] = 100,
) -> HistoryResponse:
    _validate_user_id(user_id)
    logger.info("[%s] GET /chat/%s/history | limit=%d", request_id, user_id, limit)

    try:
        messages = repo.get_messages(user_id, limit=limit)
        sessions = repo.get_sessions(user_id)

        return HistoryResponse(
            user_id=user_id,
            total_messages=len(messages),
            sessions=sessions,
            messages=[
                MessageRecord(
                    id=m.id,
                    session_id=m.session_id,
                    role=m.role,
                    content=m.content,
                    timestamp=m.timestamp,
                )
                for m in messages
            ],
        )
    except Exception as exc:
        logger.exception("[%s] History fetch failed | user=%s", request_id, user_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": "Failed to retrieve history.", "request_id": request_id},
        ) from exc


# ── DELETE /chat/{user_id}/memory ─────────────────────────────────────────────


@router.delete(
    "/{user_id}/memory",
    response_model=MemoryDeleteResponse,
    status_code=status.HTTP_200_OK,
    summary="GDPR-style reset — wipe all memory for a user",
    description=(
        "Permanently deletes all messages, sessions, eval results, flags, and "
        "compressed summaries for the given user."
    ),
    responses={
        status.HTTP_200_OK: {"description": "Memory cleared successfully"},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Invalid user_id"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Deletion failed"},
    },
)
async def delete_memory(
    user_id: UserIdPath,
    request_id: Annotated[str, Depends(get_request_id)],
    repo: AbstractMemoryRepository = Depends(get_memory_repo),
) -> MemoryDeleteResponse:
    _validate_user_id(user_id)
    logger.info("[%s] DELETE /chat/%s/memory", request_id, user_id)

    try:
        deleted_msgs, deleted_sessions = repo.clear_user_memory(user_id)
        return MemoryDeleteResponse(
            user_id=user_id,
            deleted_messages=deleted_msgs,
            deleted_sessions=deleted_sessions,
            status="memory_cleared",
        )
    except Exception as exc:
        logger.exception("[%s] Memory delete failed | user=%s", request_id, user_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": "Failed to clear user memory.", "request_id": request_id},
        ) from exc


# ── GET /chat/{user_id}/evals ─────────────────────────────────────────────────


@router.get(
    "/{user_id}/evals",
    response_model=EvalSummary,
    status_code=status.HTTP_200_OK,
    summary="Aggregated eval scores across all sessions for a user",
    description=(
        "Returns aggregate self-evaluation metrics: averages, flagged count, "
        "and percentage of high-confidence responses (confidence ≥ 0.80)."
    ),
    responses={
        status.HTTP_200_OK: {"description": "Eval summary (zeros when no evals exist)"},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Invalid user_id"},
    },
)
async def get_evals(
    user_id: UserIdPath,
    request_id: Annotated[str, Depends(get_request_id)],
    repo: AbstractMemoryRepository = Depends(get_memory_repo),
) -> EvalSummary:
    _validate_user_id(user_id)
    logger.info("[%s] GET /chat/%s/evals", request_id, user_id)

    try:
        evals = repo.get_evals(user_id)

        if not evals:
            return EvalSummary(
                total_responses=0,
                flagged_count=0,
                avg_groundedness=0.0,
                avg_relevance=0.0,
                avg_confidence=0.0,
                high_confidence_pct=0.0,
            )

        total = len(evals)
        flagged = sum(1 for e in evals if e.flagged)
        high_conf = sum(1 for e in evals if e.confidence >= 0.80)

        return EvalSummary(
            total_responses=total,
            flagged_count=flagged,
            avg_groundedness=round(sum(e.groundedness for e in evals) / total, 4),
            avg_relevance=round(sum(e.relevance for e in evals) / total, 4),
            avg_confidence=round(sum(e.confidence for e in evals) / total, 4),
            high_confidence_pct=round(high_conf / total * 100, 2),
        )
    except Exception as exc:
        logger.exception("[%s] Evals fetch failed | user=%s", request_id, user_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": "Failed to retrieve eval summary.", "request_id": request_id},
        ) from exc


# ── Helpers ───────────────────────────────────────────────────────────────────


def _validate_user_id(user_id: str) -> None:
    if not _USER_ID_PATTERN.match(user_id):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "Invalid user_id. Use only letters, numbers, underscores, and hyphens "
                f"(received: {user_id!r})."
            ),
        )
