"""
Memory factory / DI binding.

To swap the memory backend, change *only* this file (or set MEMORY_BACKEND in .env):

    MEMORY_BACKEND=sqlite   # default — SQLAlchemy (SQLite or Postgres via DATABASE_URL)
    MEMORY_BACKEND=postgres # future: PostgresMemoryRepository(db)
    MEMORY_BACKEND=mem0     # future: Mem0MemoryRepository()

Nothing in agents/, services/, or api/ needs to change.
"""
from __future__ import annotations

import logging
from typing import Literal

from sqlalchemy.orm import Session

from app.config import get_settings
from app.memory.base import AbstractMemoryRepository
from app.memory.sqlite_repository import SQLiteMemoryRepository

logger = logging.getLogger(__name__)

MemoryBackend = Literal["sqlite", "postgres", "mem0"]


def create_memory_repository(
    db: Session,
    backend: MemoryBackend | None = None,
) -> AbstractMemoryRepository:
    """Instantiate the configured memory repository."""
    selected = backend or _resolved_backend()

    if selected == "sqlite":
        return SQLiteMemoryRepository(db)

    if selected == "postgres":
        # Postgres uses the same SQLAlchemy repository — only DATABASE_URL changes.
        logger.info("Using SQLAlchemy memory repository with Postgres DATABASE_URL")
        return SQLiteMemoryRepository(db)

    if selected == "mem0":
        raise NotImplementedError(
            "Mem0 backend is not implemented yet. "
            "Create app/memory/mem0_repository.py and register it here."
        )

    raise ValueError(f"Unsupported MEMORY_BACKEND={selected!r}")


def get_memory_repository(db: Session) -> AbstractMemoryRepository:
    """FastAPI dependency — return the active memory backend."""
    return create_memory_repository(db)


def _resolved_backend() -> MemoryBackend:
    backend = getattr(get_settings(), "MEMORY_BACKEND", "sqlite").lower()
    if backend not in ("sqlite", "postgres", "mem0"):
        logger.warning("Unknown MEMORY_BACKEND=%r — falling back to sqlite", backend)
        return "sqlite"
    return backend  # type: ignore[return-value]
