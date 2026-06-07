"""
Flags API route — human-review queue for low-confidence agent responses.
"""
from __future__ import annotations

import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.memory.base import AbstractMemoryRepository
from app.memory.factory import get_memory_repository
from app.models.schemas import FlagRecord, FlagsResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["flags"])


def get_request_id(request: Request) -> str:
    header_id = request.headers.get("X-Request-ID", "").strip()
    return header_id or str(uuid.uuid4())


def get_memory_repo(db: Session = Depends(get_db)) -> AbstractMemoryRepository:
    return get_memory_repository(db)


@router.get(
    "/flags",
    response_model=FlagsResponse,
    status_code=status.HTTP_200_OK,
    summary="Human-review queue — flagged low-confidence responses",
    description=(
        "Returns all escalation flags raised when agent confidence falls below "
        "the configured threshold. Filter by `resolved` and/or `user_id`."
    ),
    responses={
        status.HTTP_200_OK: {"description": "List of flags (may be empty)"},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Invalid query parameters"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Failed to retrieve flags"},
    },
)
async def get_flags(
    request_id: Annotated[str, Depends(get_request_id)],
    resolved: Annotated[
        bool | None,
        Query(description="Filter by resolved status (`true` / `false`). Omit for all."),
    ] = None,
    user_id: Annotated[
        str | None,
        Query(max_length=255, description="Filter flags to a specific user."),
    ] = None,
    repo: AbstractMemoryRepository = Depends(get_memory_repo),
) -> FlagsResponse:
    logger.info(
        "[%s] GET /flags | resolved=%s user_id=%s",
        request_id,
        resolved,
        user_id,
    )

    try:
        flags = repo.get_flags(resolved=resolved, user_id=user_id)
        unresolved = sum(1 for f in flags if not f.resolved)

        return FlagsResponse(
            total=len(flags),
            unresolved=unresolved,
            flags=[
                FlagRecord(
                    id=f.id,
                    user_id=f.user_id,
                    session_id=f.session_id,
                    reason=f.reason,
                    message_snippet=f.message_snippet,
                    confidence_score=f.confidence_score,
                    created_at=f.created_at,
                    resolved=f.resolved,
                )
                for f in flags
            ],
        )
    except Exception as exc:
        logger.exception("[%s] Failed to fetch flags", request_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": "Failed to retrieve flags.", "request_id": request_id},
        ) from exc
