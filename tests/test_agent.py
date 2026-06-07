"""
Tests for the Sales Assistant Agent.

Run with:
    pytest tests/ -v
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.orm_models import Base
from app.db.session import get_db
from app.main import app
from app.memory.factory import get_memory_repository

# ── In-memory SQLite for tests ────────────────────────────────────────────────
# StaticPool ensures FastAPI TestClient connections share one in-memory database.

TEST_DATABASE_URL = "sqlite://"

test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)


@event.listens_for(test_engine, "connect")
def _set_sqlite_pragma(dbapi_conn, _) -> None:
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

Base.metadata.create_all(bind=test_engine)


def override_get_db():
    db = TestingSession()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)


# ── Health ────────────────────────────────────────────────────────────────────

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] in ("ok", "degraded")
    assert "version" in data
    assert "uptime_seconds" in data


# ── Catalog ───────────────────────────────────────────────────────────────────

def test_get_catalog():
    r = client.get("/catalog")
    assert r.status_code == 200
    catalog = r.json()["catalog"]
    assert "plans" in catalog
    assert len(catalog["plans"]) >= 3


# ── Memory repository ─────────────────────────────────────────────────────────

def test_memory_save_and_retrieve():
    db = TestingSession()
    repo = get_memory_repository(db)

    uid = "test-user-memory"
    repo.save_message(uid, "sess-1", "user", "Hello")
    repo.save_message(uid, "sess-1", "assistant", "Hi there!")

    messages = repo.get_messages(uid)
    assert len(messages) == 2
    assert messages[0].role == "user"
    assert messages[1].role == "assistant"
    db.close()


def test_memory_cross_session():
    db = TestingSession()
    repo = get_memory_repository(db)

    uid = "test-cross-session"
    repo.save_message(uid, "sess-A", "user", "What is Enterprise pricing?")
    repo.save_message(uid, "sess-A", "assistant", "Enterprise is $499/mo.")
    # New session
    repo.save_message(uid, "sess-B", "user", "Does that include SSO?")

    messages = repo.get_messages(uid)
    assert len(messages) == 3   # all three messages visible

    sessions = repo.get_sessions(uid)
    assert "sess-A" in sessions
    assert "sess-B" in sessions
    db.close()


def test_memory_delete():
    db = TestingSession()
    repo = get_memory_repository(db)

    uid = "test-delete-user"
    repo.save_message(uid, "sess-del", "user", "Delete me")
    repo.save_message(uid, "sess-del", "assistant", "OK")

    deleted_msgs, deleted_sessions = repo.clear_user_memory(uid)
    assert deleted_msgs == 2

    messages = repo.get_messages(uid)
    assert len(messages) == 0
    db.close()


# ── History endpoint ──────────────────────────────────────────────────────────

def test_get_history_empty():
    r = client.get("/chat/no-such-user-xyz/history")
    assert r.status_code == 200
    data = r.json()
    assert data["total_messages"] == 0
    assert data["messages"] == []


# ── Delete memory endpoint ────────────────────────────────────────────────────

def test_delete_memory_endpoint():
    # Seed some data via the repo first
    db = TestingSession()
    repo = get_memory_repository(db)
    repo.save_message("gdpr-user", "s1", "user", "Forget me")
    db.close()

    r = client.delete("/chat/gdpr-user/memory")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "memory_cleared"
    assert data["user_id"] == "gdpr-user"


# ── Evals endpoint ────────────────────────────────────────────────────────────

def test_get_evals_empty():
    r = client.get("/chat/no-eval-user/evals")
    assert r.status_code == 200
    data = r.json()
    assert data["total_responses"] == 0


def test_eval_summary_calculation():
    db = TestingSession()
    repo = get_memory_repository(db)

    uid = "eval-summary-user"
    repo.save_eval(uid, "eval-summary-s1", 0.9, 0.85, 0.88, False, "Good", ["search_catalog"])
    repo.save_eval(uid, "eval-summary-s1", 0.4, 0.5, 0.45, True, "Low conf", ["search_catalog"])

    evals = repo.get_evals(uid)
    assert len(evals) == 2
    assert sum(1 for e in evals if e.flagged) == 1
    db.close()


# ── Flags endpoint ────────────────────────────────────────────────────────────

def test_get_flags_empty():
    r = client.get("/flags")
    assert r.status_code == 200
    data = r.json()
    assert "flags" in data
    assert "total" in data


# ── search_catalog tool ───────────────────────────────────────────────────────

def test_search_catalog_enterprise():
    from app.tools.search_catalog import search_catalog

    result = search_catalog("enterprise SSO audit logs")
    assert "Enterprise" in result
    assert "SSO" in result or "499" in result


def test_search_catalog_no_match():
    from app.tools.search_catalog import search_catalog

    result = search_catalog("xyzzy dragon wizard")
    assert "No direct catalog match" in result or "Starter" in result


# ── get_user_memory tool ──────────────────────────────────────────────────────

def test_get_user_memory_no_history():
    from app.tools.get_user_memory import get_user_memory

    db = TestingSession()
    repo = get_memory_repository(db)
    result = get_user_memory("brand-new-user-9999", repo)
    assert "No prior conversation" in result
    db.close()


def test_get_user_memory_with_history():
    from app.tools.get_user_memory import get_user_memory

    db = TestingSession()
    repo = get_memory_repository(db)
    uid = "memory-tool-user"
    repo.save_message(uid, "memory-tool-s1", "user", "I need enterprise features")
    repo.save_message(uid, "memory-tool-s1", "assistant", "Enterprise plan starts at $499/mo.")

    result = get_user_memory(uid, repo)
    assert "enterprise" in result.lower() or "499" in result
    db.close()
