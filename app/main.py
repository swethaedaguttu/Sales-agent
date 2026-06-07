"""
main.py — FastAPI application factory.

Start locally:
    uvicorn app.main:app --reload --port 8000
"""
from __future__ import annotations

import logging
import sys
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.agents.sales_agent import AgentError
from app.api import catalog_router, chat_router, flags_router, health_router
from app.config import get_settings
from app.db.session import init_db

# ── Logging ───────────────────────────────────────────────────────────────────

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────


def _warn_missing_llm_credentials() -> None:
    """Log a clear warning if the active LLM provider is not configured."""
    provider = settings.LLM_PROVIDER
    if provider == "groq" and not settings.GROQ_API_KEY:
        logger.warning(
            "GROQ_API_KEY is not set — copy .env.example to .env and add your key. "
            "/chat requests will fail until configured."
        )
    elif provider == "openai" and not settings.OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY is not set — /chat requests will fail until configured.")
    elif provider == "anthropic" and not settings.ANTHROPIC_API_KEY:
        logger.warning("ANTHROPIC_API_KEY is not set — /chat requests will fail until configured.")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Starting %s v%s", settings.APP_NAME, settings.APP_VERSION)
    _warn_missing_llm_credentials()
    init_db()
    yield
    logger.info("Shutting down %s", settings.APP_NAME)


# ── OpenAPI metadata ──────────────────────────────────────────────────────────

OPENAPI_TAGS = [
    {
        "name": "health",
        "description": "Liveness and dependency health checks.",
    },
    {
        "name": "catalog",
        "description": "Product and pricing catalog used by the sales agent.",
    },
    {
        "name": "chat",
        "description": (
            "Conversational endpoints with cross-session memory, tool use, "
            "and self-evaluation on every response."
        ),
    },
    {
        "name": "flags",
        "description": "Human-review queue for low-confidence agent responses.",
    },
]

API_DESCRIPTION = """
## Persistent Sales Assistant Agent

B2B SaaS sales assistant API with:

* **Cross-session memory** — persisted in SQLite/Postgres
* **Real tool calling** — `search_catalog`, `get_user_memory`, `flag_for_human`
* **Self-evaluation** — every `/chat` response includes groundedness, relevance, and confidence scores

### Correlation IDs

Pass `X-Request-ID` on any request to propagate a trace ID through logs and error responses.
"""


def create_app() -> FastAPI:
    """Application factory — used by uvicorn and tests."""
    application = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=API_DESCRIPTION,
        openapi_tags=OPENAPI_TAGS,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    _register_middleware(application)
    _register_exception_handlers(application)
    _register_routers(application)
    _register_root_route(application)

    return application


# ── Middleware ────────────────────────────────────────────────────────────────


def _register_middleware(application: FastAPI) -> None:
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", "").strip() or str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


# ── Exception handlers ────────────────────────────────────────────────────────


def _register_exception_handlers(application: FastAPI) -> None:
    @application.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        logger.warning(
            "[%s] Validation error on %s %s: %s",
            request_id,
            request.method,
            request.url.path,
            exc.errors(),
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "detail": exc.errors(),
                "request_id": request_id,
            },
        )

    @application.exception_handler(HTTPException)
    async def http_exception_handler(
        request: Request,
        exc: HTTPException,
    ) -> JSONResponse:
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        detail = exc.detail
        if isinstance(detail, dict):
            detail.setdefault("request_id", request_id)
        elif isinstance(detail, str):
            detail = {"message": detail, "request_id": request_id}
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": detail},
            headers=getattr(exc, "headers", None),
        )

    @application.exception_handler(AgentError)
    async def agent_error_handler(request: Request, exc: AgentError) -> JSONResponse:
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        error_msg = str(exc)
        status_code = (
            status.HTTP_504_GATEWAY_TIMEOUT
            if "timed out" in error_msg.lower()
            else status.HTTP_502_BAD_GATEWAY
        )
        logger.error("[%s] AgentError: %s", request_id, exc)
        return JSONResponse(
            status_code=status_code,
            content={"detail": {"message": error_msg, "request_id": request_id}},
        )

    @application.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        logger.exception(
            "[%s] Unhandled exception on %s %s",
            request_id,
            request.method,
            request.url.path,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": {
                    "message": "An unexpected error occurred. Please try again.",
                    "request_id": request_id,
                }
            },
        )


# ── Routers ───────────────────────────────────────────────────────────────────


def _register_routers(application: FastAPI) -> None:
    application.include_router(health_router)
    application.include_router(catalog_router)
    application.include_router(chat_router)
    application.include_router(flags_router)


def _register_root_route(application: FastAPI) -> None:
    @application.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        return {
            "service": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "docs": "/docs",
            "health": "/health",
        }


# ── App instance ──────────────────────────────────────────────────────────────

app = create_app()
