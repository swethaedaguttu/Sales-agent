"""
Health API route — liveness and dependency checks.
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Request, status
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db.session import check_db_connection, get_db
from app.models.schemas import HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])

_start_time = time.time()


def get_request_id(request: Request) -> str:
    header_id = request.headers.get("X-Request-ID", "").strip()
    return header_id or str(uuid.uuid4())


def get_app_settings() -> Settings:
    return get_settings()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
    description=(
        "Returns service status, version, database connectivity, configured LLM model, "
        "and uptime. Responds with HTTP 200 when healthy or degraded; "
        "HTTP 503 when the database is unreachable."
    ),
    responses={
        status.HTTP_200_OK: {"description": "Service is healthy or degraded (see `status` field)"},
    },
)
async def health(
    request_id: Annotated[str, Depends(get_request_id)],
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_app_settings),
) -> HealthResponse:
    logger.debug("[%s] GET /health", request_id)

    db_status: str = "ok"
    try:
        db.execute(text("SELECT 1"))
    except Exception as exc:
        logger.error("[%s] DB health check failed: %s", request_id, exc)
        db_status = f"error: {exc}"

    # Fallback probe when session execute fails silently.
    if db_status == "ok" and not check_db_connection():
        db_status = "error: connection check failed"

    service_status: Literal["ok", "degraded"] = "ok" if db_status == "ok" else "degraded"

    payload = HealthResponse(
        status=service_status,
        version=settings.APP_VERSION,
        db=db_status,
        model=settings.resolved_model_name(),
        uptime_seconds=round(time.time() - _start_time, 2),
    )

    logger.info(
        "[%s] Health | status=%s db=%s uptime=%.1fs",
        request_id,
        payload.status,
        payload.db,
        payload.uptime_seconds,
    )
    return payload
