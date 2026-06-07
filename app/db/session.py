"""
Database engine, session factory, and FastAPI dependency.

Swapping SQLite → Postgres requires only changing DATABASE_URL in config;
the ORM models and this module work with any SQLAlchemy-supported backend.
"""
from __future__ import annotations

import logging
from collections.abc import Generator
from functools import lru_cache

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings
from app.db.orm_models import Base

logger = logging.getLogger(__name__)

_session_factory: sessionmaker[Session] | None = None


@lru_cache
def get_engine() -> Engine:
    """Return the shared SQLAlchemy engine (lazy-initialised)."""
    settings = get_settings()

    connect_args: dict = (
        {"check_same_thread": False}
        if settings.DATABASE_URL.startswith("sqlite")
        else {}
    )

    eng = create_engine(
        settings.DATABASE_URL,
        connect_args=connect_args,
        echo=settings.DEBUG,
        pool_pre_ping=not settings.DATABASE_URL.startswith("sqlite"),
    )

    if settings.DATABASE_URL.startswith("sqlite"):

        @event.listens_for(eng, "connect")
        def _set_sqlite_pragma(dbapi_conn, _) -> None:  # noqa: ANN001
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return eng


def _get_session_factory() -> sessionmaker[Session]:
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=get_engine(),
        )
    return _session_factory


def __getattr__(name: str):  # noqa: ANN201
    """Lazy module-level exports so importing db does not require env vars."""
    if name == "engine":
        return get_engine()
    if name == "SessionLocal":
        return _get_session_factory()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def init_db() -> None:
    """Create all tables on startup."""
    logger.info("Initialising database schema …")
    Base.metadata.create_all(bind=get_engine())
    logger.info("Database ready.")


def check_db_connection() -> bool:
    """Return True if the database is reachable."""
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        logger.exception("Database connection check failed")
        return False


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency — yields a DB session and ensures it is closed."""
    db = _get_session_factory()()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
