# Persistent Sales Assistant Agent

A production-ready B2B SaaS sales assistant API with **cross-session memory**, **real tool calling**, and **self-evaluation on every response**.

Built as a take-home assignment demonstrating persistent agent memory, structured evaluation, and clean API architecture — deployable to Render or Railway in minutes.

> **Live URL:** [https://sales-agent-d1vx.onrender.com](https://sales-agent-d1vx.onrender.com)  
> **API docs:** [https://sales-agent-d1vx.onrender.com/docs](https://sales-agent-d1vx.onrender.com/docs) · **Health:** [https://sales-agent-d1vx.onrender.com/health](https://sales-agent-d1vx.onrender.com/health)

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.12 |
| API framework | FastAPI |
| ORM | SQLAlchemy 2.0 |
| Database | SQLite (default; Postgres-compatible via `DATABASE_URL`) |
| Validation | Pydantic v2 |
| LLM | **Groq** (default) · OpenAI · Anthropic Claude — swappable via `LLM_PROVIDER` |
| Deployment | Render (live) · Railway (`railway.toml`, `nixpacks.toml`) |
| Testing | Pytest |

---

## Features

| Feature | Implementation |
|---------|----------------|
| **Cross-session memory** | Messages persisted by `user_id`; `get_user_memory` tool injects history from all sessions |
| **Product catalog search** | `search_catalog` tool — keyword + fuzzy (`difflib`) search over `catalog.json` |
| **Conversation history** | `GET /chat/{user_id}/history` returns messages across all sessions |
| **Self-evaluation scoring** | `EvalService` — structured JSON scores on every `/chat` response |
| **Human review escalation** | Low-confidence responses flagged via `flag_for_human`; queryable at `GET /flags` |
| **GDPR memory deletion** | `DELETE /chat/{user_id}/memory` wipes all user data |
| **OpenAPI documentation** | Swagger UI at `/docs`, ReDoc at `/redoc` |
| **Render / Railway deployment** | Live on [Render](https://sales-agent-d1vx.onrender.com); `runtime.txt`, `Procfile`, `railway.toml` |
| **Memory compression** *(bonus)* | `MemoryService.maybe_compress()` summarises older messages when count exceeds threshold |

All agent tools are **real Python functions** called from the agent loop — not simulated inside prompts.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CLIENT (curl / frontend)                    │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ POST /chat/{user_id}
                               │ Header: X-Request-ID (optional)
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         FastAPI App (main.py)                       │
│  • CORS + request-ID middleware                                     │
│  • Structured exception handlers                                    │
│  • Lifespan: DB init on startup                                     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ Depends(get_db) → get_memory_repository()
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  api/chat.py  (Route Handler)                       │
│  Validates request, injects SalesAgent, returns ChatResponse          │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│               agents/sales_agent.py  (Agent Loop)                    │
│                                                                     │
│  1. get_user_memory(user_id, repo)  ──► tools/get_user_memory.py    │
│        ↓ memory context string                                      │
│  2. search_catalog(message)        ──► tools/search_catalog.py      │
│        ↓ catalog context string                                     │
│  3. Build system prompt (contexts injected)                         │
│  4. LLMService.complete()           ──► Groq    │
│        ↓ assistant_text                                             │
│  5. EvalService.evaluate()          ──► services/eval_service.py    │
│        ↓ EvalBlock                                                  │
│  6. MemoryService.persist_turn()    ──► save user + assistant msgs  │
│  7. MemoryService.save_eval_record() ──► persist scores             │
│  8. [if flagged] flag_for_human()   ──► tools/flag_for_human.py     │
│  9. [if >30 msgs] MemoryService.maybe_compress()                    │
│        ↓                                                            │
│  Return ChatResponse (with request_id + session_id)                 │
└─────────────────────────────────────────────────────────────────────┘
                               │
                  ┌────────────┴────────────┐
                  ▼                         ▼
        memory/sqlite_repository     services/eval_service.py
        (AbstractMemoryRepository)   (second LLM call → JSON scores)
                  │
                  ▼
          SQLite / SQLAlchemy
          ┌─────────────────┐
          │ sessions        │
          │ messages        │
          │ eval_results    │
          │ flags           │
          │ memory_summaries│
          └─────────────────┘
```

---

## Example Conversation

This example demonstrates **cross-session memory**: the agent remembers Enterprise pricing from Session A when the user asks a follow-up in Session B without re-stating context.

### Turn 1 — Enterprise plan query (Session A)

**User:** `What does your Enterprise plan include, and how much does it cost?`

**Agent:** *(searches catalog, responds with grounded pricing)*

> Our Enterprise plan is **$499/month** ($399/month billed annually) and includes unlimited users, SSO (SAML 2.0, OIDC), audit logs, dedicated SLA, 24/7 support, and data residency options (US, EU, APAC). Would you like to book a demo or start a 14-day free trial?

**Tools called:** `get_user_memory`, `search_catalog`

---

### Turn 2 — Follow-up SSO question (Session B — new `session_id`)

**User:** `Does that include SSO? And what about data residency options?`

**Agent:** *(retrieves Session A history from DB — no pricing re-sent in request body)*

> Yes — the Enterprise plan we discussed includes **SSO via SAML 2.0 and OIDC**, compatible with Okta, Azure AD, and Google Workspace. Data residency options for US, EU, and APAC are also included on Enterprise at no extra cost.

**Why this proves memory:** The user said *"that"* and *"the Enterprise plan we discussed"* without repeating $499. The agent resolved the referent from persisted memory keyed by `user_id`, not from the current request payload.

---

### Verify cross-session history

```bash
curl -s http://localhost:8000/chat/demo-user-001/history | python -m json.tool
```

The response lists messages from **two different `session_id` values** under the same `user_id`.

---


## Screenshots

Local demo at `http://127.0.0.1:8000` using Groq (`llama-3.3-70b-versatile`).

### Root Endpoint

![Root endpoint service metadata](docs/screenshots/root-endpoint.png)

*Service name, version, and links to `/docs` and `/health`.*

### Health Check

![GET /health — service health check](docs/screenshots/health-check.png)

*`GET /health` returns status, DB connectivity, model name, and uptime.*

### Swagger Documentation

![POST /chat/{user_id} in Swagger UI](docs/screenshots/swagger-chat-request.png)

*Interactive OpenAPI docs at `/docs` — testing the chat endpoint with a user message.*

### Chat Endpoint

![POST /chat response with eval block](docs/screenshots/chat-endpoint.png)

*Enterprise plan query response with `eval` scores, `tools_called`, `session_id`, and `request_id`.*

### Conversation History

![GET /chat/{user_id}/history](docs/screenshots/conversation-history.png)

*`GET /chat/{user_id}/history` showing user and assistant messages for a session.*

### Evaluation Dashboard

![GET /chat/{user_id}/evals aggregated metrics](docs/screenshots/evaluation-dashboard.png)

*Aggregated eval summary — avg groundedness, relevance, confidence, and high-confidence percentage.*

### Catalog Endpoint

![GET /catalog full product catalog](docs/screenshots/catalog-endpoint.png)

*Full NexusHQ catalog JSON with Starter, Growth, and Enterprise plans.*

### GDPR Memory Deletion

![DELETE /chat/{user_id}/memory](docs/screenshots/gdpr-memory-delete.png)

*`DELETE /chat/{user_id}/memory` wipes all messages and sessions for a user.*

### Human Review Flags

![GET /flags human-review queue](docs/screenshots/flags-endpoint.png)

*`GET /flags?user_id=...` returns the human-review escalation queue.*

---

## Endpoints

| Method | Path | Status | Description |
|--------|------|--------|-------------|
| `POST` | `/chat/{user_id}` | 200 | Send a message; receive response + self-eval |
| `GET` | `/chat/{user_id}/history` | 200 | Full conversation history (optional `limit` query) |
| `DELETE` | `/chat/{user_id}/memory` | 200 | GDPR reset — wipe all user data |
| `GET` | `/chat/{user_id}/evals` | 200 | Aggregated eval scores for a user |
| `GET` | `/catalog` | 200 | Full product/pricing catalog |
| `GET` | `/flags` | 200 | Human-review queue (`resolved`, `user_id` filters) |
| `GET` | `/health` | 200 | Service health check |
| `GET` | `/docs` | — | Swagger UI |
| `GET` | `/redoc` | — | ReDoc API docs |

Pass `X-Request-ID` on any request for log correlation; the ID is echoed in the response header and error payloads.

---

## Memory Design

### How it works

Every user message and assistant response is persisted in the `messages` table keyed by `user_id` and `session_id`. On every `/chat` request:

1. `get_user_memory(user_id, repo)` queries the DB for the last N messages (configurable via `MEMORY_MAX_MESSAGES`, default 20).
2. Any compressed summary of older messages is prepended from `memory_summaries`.
3. The full context string is injected into the system prompt.
4. Both the user message and assistant response are written back via `MemoryService.persist_turn()`.

**Cross-session continuity** is achieved because `get_recent_messages()` filters by `user_id` across *all* session IDs — not just the current one. A new session is a new UUID; the memory layer sees all of them.

### Memory layer abstraction

`memory/base.py` defines `AbstractMemoryRepository` with 12 abstract methods.
`memory/sqlite_repository.py` implements all 12 using SQLAlchemy.
`memory/factory.py` is the **single wiring point**:

```python
# Swap to Postgres (same repository, different DATABASE_URL):
MEMORY_BACKEND=postgres
DATABASE_URL=postgresql://user:pass@host:5432/sales_agent

# Swap to Mem0 (future — not yet implemented):
from app.memory.mem0_repository import Mem0MemoryRepository
return Mem0MemoryRepository()
```

Zero changes in `agents/`, `services/`, or `api/`.

### Memory compression (bonus)

When a user accumulates more than `MEMORY_SUMMARY_THRESHOLD` (default 30) messages, `MemoryService.maybe_compress()` (alias: `MemoryCompressionService`) compresses the oldest half into a 3–5 sentence summary stored in `memory_summaries`. Recent messages stay verbatim. This bounds token costs as conversations grow.

### At scale — what we'd use instead

| Scale | Recommended backend |
|-------|---------------------|
| MVP / single-node | SQLite (current) |
| Multi-instance production | PostgreSQL + pgvector |
| Sub-millisecond retrieval | Redis (recent turns) + Postgres (long-term) |
| Managed semantic memory | [Mem0](https://mem0.ai) — drop-in via `memory/factory.py` |

---

## Eval Design

### How it works

After every agent response, `EvalService` makes a **second LLM call** with a structured JSON evaluation prompt. The evaluator scores three dimensions:

| Dimension | What it measures |
|-----------|------------------|
| `groundedness` | Are all claims traceable to catalog/memory context? |
| `relevance` | Does the response directly answer the user's question? |
| `confidence` | Overall reliability — penalises vagueness and speculation |

If `confidence < EVAL_FLAG_THRESHOLD` (default 0.60), any dimension falls below threshold, or the evaluator sets `flagged: true`, then `flag_for_human()` persists an entry in the `flags` table for human review.

Every eval result is stored in `eval_results` and queryable via `GET /chat/{user_id}/evals`.

### Limitations

- **Self-reporting bias**: The same model family that generated the response also scores it. It can be optimistically biased.
- **No ground-truth calibration**: Scores aren't calibrated against human ratings.
- **Latency cost**: A second API call adds ~500–800 ms per response.

### What we'd replace it with at scale

- **Separate judge model**: Use GPT-4 or Prometheus-7B as an independent evaluator.
- **LLM-as-a-judge with rubrics**: [RAGAS](https://github.com/explodinggradients/ragas) or [DeepEval](https://github.com/confident-ai/deepeval) frameworks.
- **Human-in-the-loop calibration**: Sample flagged responses for human review; feed back into score calibration.

---

## Project Structure

```
sales-agent/
├── app/
│   ├── api/
│   │   ├── chat.py          # POST /chat, GET /history, DELETE /memory, GET /evals
│   │   ├── catalog.py       # GET /catalog
│   │   ├── flags.py         # GET /flags
│   │   └── health.py        # GET /health
│   ├── agents/
│   │   └── sales_agent.py   # Full agent loop (memory → catalog → LLM → eval → persist)
│   ├── memory/
│   │   ├── base.py          # AbstractMemoryRepository interface
│   │   ├── sqlite_repository.py  # SQLAlchemy implementation
│   │   └── factory.py       # ← Only file to change when swapping backends
│   ├── tools/
│   │   ├── search_catalog.py     # Keyword + fuzzy catalog search
│   │   ├── get_user_memory.py    # Queries DB memory for context
│   │   └── flag_for_human.py     # Persists escalation flags
│   ├── services/
│   │   ├── eval_service.py       # EvalService — LLM self-evaluation → EvalBlock
│   │   ├── memory_service.py     # MemoryService — persistence + compression
│   │   └── llm/                  # Swappable LLM layer (factory, service, Groq/OpenAI/Anthropic providers)
│   ├── models/
│   │   └── schemas.py       # Pydantic request/response models
│   ├── db/
│   │   ├── orm_models.py    # SQLAlchemy ORM models (sessions, messages, evals, flags, summaries)
│   │   └── session.py       # Engine, SessionLocal, get_db dependency
│   ├── config.py            # Settings (pydantic-settings, .env)
│   ├── catalog.json         # Product/pricing catalog
│   └── main.py              # FastAPI app factory + lifespan
├── tests/
│   └── test_agent.py        # Pytest test suite (14 tests)
├── docs/
│   └── screenshots/         # API endpoint screenshots (referenced in README)
├── .env.example
├── .gitignore
├── runtime.txt              # Render Python 3.12.8 pin
├── nixpacks.toml            # Railway Python 3.12 pin
├── Procfile
├── railway.toml
├── requirements.txt
└── README.md
```

---

## Local Setup

```bash
# 1. Clone and enter directory
git clone https://github.com/your-org/sales-agent.git
cd sales-agent

# 2. Create virtual environment (Python 3.12 recommended)
python3.12 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# 5. Start the server
uvicorn app.main:app --reload --port 8000

# 6. Run tests
pytest tests/ -v
```

Open http://localhost:8000/docs for Swagger UI.

---

## Render Deployment (Live)

**Production URL:** [https://sales-agent-d1vx.onrender.com](https://sales-agent-d1vx.onrender.com)

| Setting | Value |
|---------|--------|
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `uvicorn app.main:app --host 0.0.0.0 --port $PORT` |
| **Python version** | `3.12.8` (via `runtime.txt`) |

**Required environment variables:** `GROQ_API_KEY`, `LLM_PROVIDER=groq`, `MODEL_NAME=llama-3.3-70b-versatile` (see [Environment Variables](#environment-variables)).

Quick smoke test:

```bash
curl -s https://sales-agent-d1vx.onrender.com/health | python -m json.tool
curl -s https://sales-agent-d1vx.onrender.com/catalog | python -m json.tool
```

> **SQLite on Render:** The free tier uses ephemeral storage — the database resets on redeploy. For persistent memory, attach a [Render PostgreSQL](https://render.com/docs/databases) instance and set `DATABASE_URL` + `MEMORY_BACKEND=postgres`.

---

## Railway Deployment

```bash
# 1. Install Railway CLI
npm install -g @railway/cli

# 2. Login and initialise
railway login
railway init

# 3. Set environment variables
railway variables set GROQ_API_KEY=gsk-...
railway variables set LLM_PROVIDER=groq
railway variables set MODEL_NAME=llama-3.3-70b-versatile

# 4. Deploy
railway up

# 5. Get your public URL
railway open
```

The `railway.toml` and `nixpacks.toml` files handle Python 3.12 and the uvicorn start command automatically. Health checks probe `/health`.

> **SQLite on Railway:** Railway has an ephemeral filesystem — the SQLite DB resets on each deploy. For persistent storage, provision a PostgreSQL plugin, set `DATABASE_URL` to the Postgres URL, and set `MEMORY_BACKEND=postgres`.

---

## Cross-Session Memory Demo (curl)

Live demo against [https://sales-agent-d1vx.onrender.com](https://sales-agent-d1vx.onrender.com) (or use `http://localhost:8000` locally).

### Call 1 — Establish context (Session A)

```bash
curl -s -X POST https://sales-agent-d1vx.onrender.com/chat/demo-user-001 \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: demo-call-1" \
  -d '{"message": "What does your Enterprise plan include, and how much does it cost?"}' \
  | python -m json.tool
```

**Expected response excerpt:**

```json
{
  "response": "Our Enterprise plan is $499/month (or $399/month billed annually)...",
  "eval": {
    "groundedness": 0.92,
    "relevance": 0.95,
    "confidence": 0.90,
    "flagged": false,
    "reasoning": "Response sourced directly from catalog. All claims verifiable."
  },
  "tools_called": ["get_user_memory", "search_catalog"],
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "user_id": "demo-user-001",
  "request_id": "demo-call-1",
  "timestamp": "2026-06-06T10:30:00"
}
```

### Call 2 — New session, uses prior context (Session B)

```bash
curl -s -X POST https://sales-agent-d1vx.onrender.com/chat/demo-user-001 \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: demo-call-2" \
  -d '{"message": "Does that include SSO? And what about data residency options?"}' \
  | python -m json.tool
```

The agent answers **without re-sending pricing context** because `get_user_memory` retrieved it from the DB. The `session_id` will be a *different* UUID — proving cross-session continuity.

### Verify history spans both sessions

```bash
curl -s https://sales-agent-d1vx.onrender.com/chat/demo-user-001/history \
  | python -m json.tool
```

You will see messages from both session UUIDs under the same `user_id`.

### Other useful commands

```bash
# Aggregated eval scores
curl -s https://sales-agent-d1vx.onrender.com/chat/demo-user-001/evals | python -m json.tool

# Human-review flags
curl -s https://sales-agent-d1vx.onrender.com/flags | python -m json.tool

# GDPR reset
curl -s -X DELETE https://sales-agent-d1vx.onrender.com/chat/demo-user-001/memory | python -m json.tool

# Full catalog
curl -s https://sales-agent-d1vx.onrender.com/catalog | python -m json.tool

# Health check
curl -s https://sales-agent-d1vx.onrender.com/health | python -m json.tool
```

---

## Response Schema

```jsonc
{
  "response": "Our Enterprise plan is $499/month...",
  "eval": {
    "groundedness": 0.92,   // Claims traceable to catalog/memory?
    "relevance":    0.95,   // Directly answers the question?
    "confidence":   0.90,   // Overall reliability
    "flagged":      false,  // true if confidence < 0.60 or speculation detected
    "reasoning":    "Response sourced directly from catalog."
  },
  "tools_called": ["get_user_memory", "search_catalog"],
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "user_id": "demo-user-001",
  "request_id": "550e8400-e29b-41d4-a716-446655440001",
  "timestamp": "2026-06-06T10:30:00"
}
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | *(required for Groq)* | Groq API key |
| `LLM_PROVIDER` | `groq` | LLM backend: `groq` \| `openai` \| `anthropic` |
| `MODEL_NAME` | `llama-3.3-70b-versatile` | Model identifier for the active provider |
| `LLM_TIMEOUT_SECONDS` | `60` | LLM request timeout |
| `OPENAI_API_KEY` | — | Required when `LLM_PROVIDER=openai` |
| `ANTHROPIC_API_KEY` | — | Required when `LLM_PROVIDER=anthropic` |
| `DATABASE_URL` | `sqlite:///./sales_agent.db` | DB connection string |
| `MEMORY_BACKEND` | `sqlite` | Memory backend: `sqlite` \| `postgres` \| `mem0` |
| `AGENT_MAX_TOKENS` | `2048` | Max tokens per LLM response |
| `MEMORY_MAX_MESSAGES` | `20` | Max messages injected as context |
| `MEMORY_SUMMARY_THRESHOLD` | `30` | Compress memory after N messages |
| `EVAL_FLAG_THRESHOLD` | `0.60` | Flag response if confidence below this |
| `LOG_LEVEL` | `INFO` | `DEBUG` \| `INFO` \| `WARNING` |
| `DEBUG` | `false` | Enable SQLAlchemy query logging |
| `CATALOG_PATH` | `app/catalog.json` | Path to product catalog JSON |

---

## Design Decisions & Tradeoffs

### Why SQLite?

Zero-dependency, zero-config for development and single-instance deployments. The memory abstraction means migrating to Postgres requires changing `DATABASE_URL` and `MEMORY_BACKEND` — no business-logic changes.

### Why keyword + fuzzy search for catalog?

Fast, zero-dependency, and transparent. `search_catalog` combines token overlap with `difflib` fuzzy matching. The scoring layer is isolated in `tools/search_catalog.py` — swap for pgvector or a vector store without touching the agent.

### Why a second LLM call for eval?

Simplest implementation with the most explainability. The alternative — a fine-tuned reward model — provides better calibration but requires labelled data and another model to maintain. The second-call approach catches gross hallucinations on day one.

### Why store full message history instead of embeddings?

History stays human-readable and auditable. `MemoryService.maybe_compress()` caps token cost as conversations grow. Embeddings would be the next step for semantic retrieval at scale.

### Why a swappable LLM layer?

`services/llm/` exposes a provider-agnostic `LLMService` facade. Groq is the default (OpenAI-compatible client); OpenAI and Anthropic are one env-var change away. Business logic never imports provider SDKs directly.

---

## Future Improvements

| Area | Planned enhancement |
|------|---------------------|
| Database | Dedicated PostgreSQL backend with connection pooling |
| Search | pgvector semantic search over catalog and conversation history |
| Caching | Redis for hot session context and catalog query results |
| RAG | Document ingestion pipeline for sales collateral beyond static JSON |
| Architecture | Multi-agent routing (qualification vs. pricing vs. support specialists) |
| Evaluation | Independent judge model (separate from the generation model) |
| Security | Authentication, API keys, and role-based access control (RBAC) |
| Memory | Mem0 integration for managed long-term semantic memory |

---

## Project Status

### Current implementation status

| Component | Status |
|-----------|--------|
| Cross-session memory (`AbstractMemoryRepository`) | ✅ Complete |
| Memory compression (`MemoryService` / `MemoryCompressionService`) | ✅ Complete |
| Catalog search tool (keyword + fuzzy) | ✅ Complete |
| Self-evaluation (`EvalService`) | ✅ Complete |
| Human review flagging (`flag_for_human` + `GET /flags`) | ✅ Complete |
| GDPR deletion (`DELETE /chat/{user_id}/memory`) | ✅ Complete |
| Swappable LLM layer (Groq / OpenAI / Anthropic) | ✅ Complete |
| Mem0 memory backend | ⏳ Not implemented (factory stub) |
| Authentication / RBAC | ⏳ Not implemented |

### Tested endpoints

All endpoints are covered by the Pytest suite (`tests/test_agent.py` — **14 tests passing**):

- `GET /health`
- `GET /catalog`
- `POST /chat/{user_id}` *(agent flow tested via memory/tools; LLM mocked in unit tests)*
- `GET /chat/{user_id}/history`
- `DELETE /chat/{user_id}/memory`
- `GET /chat/{user_id}/evals`
- `GET /flags`
- Memory repository (save, cross-session, delete)
- Tools: `search_catalog`, `get_user_memory`

Run locally:

```bash
pytest tests/ -v
```

### Deployment readiness

| Requirement | Status |
|-------------|--------|
| **Live deployment** | ✅ [https://sales-agent-d1vx.onrender.com](https://sales-agent-d1vx.onrender.com) |
| Render config (`runtime.txt`, `Procfile`) | ✅ Ready |
| Railway config (`nixpacks.toml`, `railway.toml`) | ✅ Ready |
| Health check endpoint | ✅ `/health` |
| Environment variable documentation | ✅ `.env.example` |
| OpenAPI docs | ✅ `/docs`, `/redoc` |
| Persistent DB on Render | ✅ Postgres-ready (`DATABASE_URL` + `MEMORY_BACKEND=postgres`); default SQLite is ephemeral on redeploy |
| Live demo curl commands | ✅ Documented above |

---

## License

This project is licensed under the MIT License.
