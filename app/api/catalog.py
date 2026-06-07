"""
Catalog API route — exposes the full product/pricing catalog.
"""
from __future__ import annotations

import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.models.schemas import CatalogResponse
from app.tools.search_catalog import get_full_catalog

logger = logging.getLogger(__name__)

router = APIRouter(tags=["catalog"])


def get_request_id(request: Request) -> str:
    header_id = request.headers.get("X-Request-ID", "").strip()
    return header_id or str(uuid.uuid4())


@router.get(
    "/catalog",
    response_model=CatalogResponse,
    status_code=status.HTTP_200_OK,
    summary="Returns the full product/pricing catalog",
    description=(
        "Returns the complete NexusHQ catalog JSON including plans, add-ons, FAQs, "
        "and contact information. This is the same data source used by the "
        "`search_catalog` agent tool."
    ),
    responses={
        status.HTTP_200_OK: {"description": "Full catalog document"},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"description": "Catalog file missing or unreadable"},
    },
)
async def get_catalog(
    request_id: Annotated[str, Depends(get_request_id)],
) -> CatalogResponse:
    logger.info("[%s] GET /catalog", request_id)

    try:
        catalog = get_full_catalog()
        if not catalog.get("plans"):
            logger.warning("[%s] Catalog loaded but contains no plans", request_id)
        return CatalogResponse(catalog=catalog)
    except FileNotFoundError as exc:
        logger.error("[%s] Catalog file not found: %s", request_id, exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "message": "Catalog file not found.",
                "request_id": request_id,
            },
        ) from exc
    except Exception as exc:
        logger.exception("[%s] Failed to load catalog", request_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "message": "Failed to load catalog.",
                "request_id": request_id,
            },
        ) from exc
