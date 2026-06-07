"""Shared JSON extraction helpers for LLM responses."""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.services.llm.exceptions import LLMJSONParseError

logger = logging.getLogger(__name__)

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)


def strip_markdown_fences(text: str) -> str:
    """Remove optional ```json fences from model output."""
    stripped = text.strip()
    match = _JSON_FENCE_RE.search(stripped)
    if match:
        return match.group(1).strip()
    return re.sub(r"^```[a-z]*\n?", "", stripped, flags=re.MULTILINE).removesuffix("```").strip()


def parse_json_object(raw: str) -> dict[str, Any]:
    """
    Parse a JSON object from raw LLM text.

    Tries direct parse first, then fenced extraction, then first {...} block.
    """
    candidates = [raw.strip(), strip_markdown_fences(raw)]

    brace_match = re.search(r"\{[\s\S]*\}", raw)
    if brace_match:
        candidates.append(brace_match.group(0))

    last_error: Exception | None = None
    for candidate in candidates:
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
            raise LLMJSONParseError(f"Expected JSON object, got {type(data).__name__}")
        except (json.JSONDecodeError, LLMJSONParseError) as exc:
            last_error = exc
            continue

    logger.error("Failed to parse JSON from LLM output: %s", raw[:500])
    raise LLMJSONParseError(f"Could not parse JSON object: {last_error}")
