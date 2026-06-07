from app.db.orm_models import (
    Base,
    ChatSession,
    EvalResult,
    Flag,
    MemorySummary,
    Message,
)
from app.db.session import check_db_connection, get_db, get_engine, init_db

__all__ = [
    "Base",
    "ChatSession",
    "EvalResult",
    "Flag",
    "MemorySummary",
    "Message",
    "SessionLocal",
    "check_db_connection",
    "engine",
    "get_db",
    "get_engine",
    "init_db",
]


def __getattr__(name: str):  # noqa: ANN201
    """Lazy exports — avoid loading DB engine (and settings) on package import."""
    if name in ("SessionLocal", "engine"):
        from app.db import session

        return getattr(session, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
