"""
search_catalog tool — keyword + fuzzy search over catalog.json.

Real callable function used by the agent loop (not prompt simulation).

Architecture note: the CatalogSearcher class is structured so that replacing
keyword/fuzzy matching with a vector store only requires overriding `_score_entry`.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Literal

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

CatalogEntryType = Literal["plan", "faq", "addon", "contact"]

# Minimum combined score to include a result (0–1).
_MIN_SCORE = 0.20
# When there is zero keyword overlap, require strong fuzzy signal (typos/near-names).
_MIN_FUZZY_WITHOUT_KEYWORD = 0.52
_TOP_K = 5

# ── Structured result types ───────────────────────────────────────────────────


@dataclass(frozen=True)
class CatalogMatch:
    """A single ranked catalog hit."""

    entry_type: CatalogEntryType
    name: str
    score: float
    keyword_score: float
    fuzzy_score: float
    summary: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CatalogSearchResponse:
    """Structured output from search_catalog."""

    query: str
    match_count: int
    matches: list[CatalogMatch]
    fallback: bool = False

    def to_context_string(self) -> str:
        """Format for LLM / agent context injection."""
        if self.fallback or not self.matches:
            return (
                "No direct catalog match found for that query. "
                "Available plans: Starter ($49/mo), Growth ($199/mo), Enterprise ($499/mo)."
            )

        lines = [f"=== Catalog Search Results ({self.match_count} match(es)) ==="]
        lines.append(f'Query: "{self.query}"')
        lines.append("")

        for index, match in enumerate(self.matches, start=1):
            lines.append(
                f"[{index}] {match.entry_type.upper()} | score={match.score:.2f} "
                f"(keyword={match.keyword_score:.2f}, fuzzy={match.fuzzy_score:.2f}) "
                f"| {match.name}"
            )
            lines.append(match.summary)
            lines.append("")

        return "\n".join(lines).strip()


# ── Catalog loader ────────────────────────────────────────────────────────────

_catalog_cache: dict[str, Any] | None = None


def _load_catalog() -> dict[str, Any]:
    global _catalog_cache
    if _catalog_cache is None:
        path = Path(settings.CATALOG_PATH)
        if not path.exists():
            path = Path(__file__).parent.parent / "catalog.json"
        if not path.exists():
            raise FileNotFoundError(f"Catalog not found at {settings.CATALOG_PATH}")
        with path.open(encoding="utf-8") as handle:
            _catalog_cache = json.load(handle)
        logger.info("Catalog loaded from %s", path)
    return _catalog_cache


def get_full_catalog() -> dict[str, Any]:
    """Return the full parsed catalog document."""
    return _load_catalog()


def reload_catalog() -> dict[str, Any]:
    """Clear cache and reload catalog from disk (useful in tests)."""
    global _catalog_cache
    _catalog_cache = None
    return _load_catalog()


# ── Scoring ───────────────────────────────────────────────────────────────────


def _tokenise(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def _keyword_score(query_tokens: set[str], text: str) -> float:
    """Token-overlap score in [0, 1]."""
    if not query_tokens:
        return 0.0
    entry_tokens = _tokenise(text)
    if not entry_tokens:
        return 0.0
    overlap = query_tokens & entry_tokens
    return len(overlap) / max(len(query_tokens), len(entry_tokens))


def _fuzzy_score(query: str, text: str) -> float:
    """
    Fuzzy similarity in [0, 1] using difflib.SequenceMatcher.

    Combines whole-string ratio with per-token best partial matches.
    """
    query_clean = query.lower().strip()
    text_clean = text.lower()
    if not query_clean or not text_clean:
        return 0.0

    whole_ratio = SequenceMatcher(None, query_clean, text_clean).ratio()

    query_tokens = _tokenise(query)
    text_tokens = _tokenise(text)
    if not query_tokens or not text_tokens:
        return whole_ratio

    token_scores: list[float] = []
    for q_token in query_tokens:
        best = max(SequenceMatcher(None, q_token, t_token).ratio() for t_token in text_tokens)
        token_scores.append(best)

    token_ratio = sum(token_scores) / len(token_scores)

    # Phrase-level partial match (e.g. "audit log" vs "audit logs")
    partial = SequenceMatcher(None, query_clean, text_clean).quick_ratio()

    return max(whole_ratio, token_ratio, partial)


def _combined_score(keyword: float, fuzzy: float) -> float:
    """Weighted blend — keyword primary, fuzzy catches typos and near-matches."""
    if keyword == 0.0 and fuzzy == 0.0:
        return 0.0
    return round(0.55 * keyword + 0.45 * fuzzy, 4)


def _is_relevant_match(keyword: float, fuzzy: float, combined: float) -> bool:
    """Filter out low-confidence fuzzy noise on unrelated queries."""
    if combined < _MIN_SCORE:
        return False
    if keyword == 0.0 and fuzzy < _MIN_FUZZY_WITHOUT_KEYWORD:
        return False
    return True


def _plan_summary(plan: dict[str, Any]) -> str:
    features = "\n    • ".join(plan.get("features", []))
    not_inc = plan.get("not_included", [])
    not_inc_str = "\n    ✗ " + "\n    ✗ ".join(not_inc) if not_inc else ""
    return (
        f"  Price: {plan['price']} (annual: {plan.get('annual_price', 'N/A')})\n"
        f"  Users: {plan.get('users', 'N/A')}\n"
        f"  Included:\n    • {features}{not_inc_str}"
    )


def _iter_catalog_entries(catalog: dict[str, Any]) -> list[tuple[CatalogEntryType, str, str, dict[str, Any]]]:
    """Yield (type, name, searchable_text, raw_entry) for every catalog item."""
    entries: list[tuple[CatalogEntryType, str, str, dict[str, Any]]] = []

    for plan in catalog.get("plans", []):
        text = (
            f"{plan['name']} {plan['price']} {plan.get('annual_price', '')} "
            f"{plan.get('users', '')} "
            f"{' '.join(plan.get('features', []))} "
            f"{' '.join(plan.get('not_included', []))}"
        )
        entries.append(("plan", plan["name"], text, plan))

    for faq in catalog.get("faqs", []):
        text = f"{faq['question']} {faq['answer']}"
        entries.append(("faq", faq["question"], text, faq))

    for addon in catalog.get("addons", []):
        text = f"{addon['name']} {addon['price']}"
        entries.append(("addon", addon["name"], text, addon))

    contact = catalog.get("contact", {})
    if contact:
        text = " ".join(str(v) for v in contact.values())
        entries.append(("contact", "Contact", text, contact))

    return entries


def _build_match(
    entry_type: CatalogEntryType,
    name: str,
    raw: dict[str, Any],
    keyword_score: float,
    fuzzy_score: float,
    combined: float,
) -> CatalogMatch:
    if entry_type == "plan":
        summary = _plan_summary(raw)
        details = {
            "price": raw.get("price"),
            "annual_price": raw.get("annual_price"),
            "users": raw.get("users"),
            "features": raw.get("features", []),
            "not_included": raw.get("not_included", []),
        }
    elif entry_type == "faq":
        summary = f"  Q: {raw['question']}\n  A: {raw['answer']}"
        details = {"question": raw["question"], "answer": raw["answer"]}
    elif entry_type == "addon":
        summary = f"  Price: {raw['price']}"
        details = {"price": raw.get("price")}
    else:
        summary = "\n".join(f"  {key}: {value}" for key, value in raw.items())
        details = dict(raw)

    return CatalogMatch(
        entry_type=entry_type,
        name=name,
        score=combined,
        keyword_score=round(keyword_score, 4),
        fuzzy_score=round(fuzzy_score, 4),
        summary=summary,
        details=details,
    )


# ── Public tool API ───────────────────────────────────────────────────────────


def search_catalog_structured(query: str) -> CatalogSearchResponse:
    """
    Search the product catalog and return structured ranked results.

    Uses keyword token overlap plus difflib fuzzy matching.
    """
    query = query.strip()
    if not query:
        return CatalogSearchResponse(query=query, match_count=0, matches=[], fallback=True)

    logger.info("search_catalog_structured | query=%r", query)

    catalog = _load_catalog()
    query_tokens = _tokenise(query)
    scored: list[CatalogMatch] = []

    for entry_type, name, text, raw in _iter_catalog_entries(catalog):
        kw = _keyword_score(query_tokens, text)
        fz = _fuzzy_score(query, text)
        combined = _combined_score(kw, fz)

        if _is_relevant_match(kw, fz, combined):
            scored.append(_build_match(entry_type, name, raw, kw, fz, combined))

    scored.sort(key=lambda m: m.score, reverse=True)
    top = scored[:_TOP_K]

    if not top:
        logger.info("search_catalog: no results for query=%r", query)
        return CatalogSearchResponse(query=query, match_count=0, matches=[], fallback=True)

    logger.info("search_catalog: %d result(s) for query=%r", len(top), query)
    return CatalogSearchResponse(query=query, match_count=len(top), matches=top)


def search_catalog(query: str) -> str:
    """
    Search the product catalog for information relevant to *query*.

    Returns a formatted context string for the agent (backed by structured search).
    """
    logger.info("search_catalog called | query=%r", query)
    response = search_catalog_structured(query)
    return response.to_context_string()
