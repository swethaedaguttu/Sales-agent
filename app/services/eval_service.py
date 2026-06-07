"""
EvalService — self-scores every assistant response via a structured LLM call.

Design: a second, small LLM call is made immediately after the agent generates
its response.  The prompt forces JSON output with four fields.

Limitations:
• Self-reported scores can be optimistically biased.
• A separate reward/critique model would be more reliable at scale.
• For now this catches gross hallucinations and out-of-scope responses.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from app.config import get_settings
from app.models.schemas import EvalBlock
from app.services.llm import LLMError, LLMService, get_llm_service

logger = logging.getLogger(__name__)
settings = get_settings()

_EVAL_SYSTEM = """\
You are a strict quality evaluator for a B2B SaaS sales assistant.
You will receive a user question, the assistant's response, and the catalog
context that was available.  Score the response on three dimensions and return
ONLY valid JSON — no markdown, no explanation outside the JSON.

JSON schema:
{
  "groundedness": <float 0-1>,
  "relevance": <float 0-1>,
  "confidence": <float 0-1>,
  "flagged": <bool>,
  "reasoning": <string>
}

Rules:
- groundedness: Is every claim traceable to the catalog context?
- relevance: Does the response directly answer the user question?
- confidence: Overall reliability; penalise vagueness or speculation
- flagged: true if any score < 0.60 OR response contains speculation
- reasoning: at most 2 sentences explaining the scores
"""


@dataclass(frozen=True)
class EvalRequest:
    """Input payload for a single self-evaluation."""

    user_message: str
    assistant_response: str
    catalog_context: str
    memory_context: str
    user_id: str
    session_id: str
    request_id: str


class EvalService:
    """Runs structured self-evaluation on assistant responses."""

    def __init__(self, llm: LLMService | None = None) -> None:
        self._llm = llm or get_llm_service()

    def evaluate(self, request: EvalRequest) -> EvalBlock:
        """
        Run self-evaluation and return a structured EvalBlock.

        Never raises — on any failure returns a safe default flagged block.
        """
        logger.info(
            "[%s] EvalService.evaluate | user=%s session=%s",
            request.request_id,
            request.user_id,
            request.session_id,
        )

        eval_prompt = (
            f"USER QUESTION:\n{request.user_message}\n\n"
            f"CATALOG CONTEXT AVAILABLE:\n{request.catalog_context}\n\n"
            f"MEMORY CONTEXT AVAILABLE:\n{request.memory_context}\n\n"
            f"ASSISTANT RESPONSE:\n{request.assistant_response}\n\n"
            "Score the response now."
        )

        try:
            data = self._llm.complete_json(
                system=_EVAL_SYSTEM,
                user_message=eval_prompt,
                max_tokens=400,
                temperature=0.0,
            )
            block = self._parse_eval_data(data)

            logger.info(
                "[%s] Eval complete | user=%s session=%s G=%.2f R=%.2f C=%.2f flagged=%s",
                request.request_id,
                request.user_id,
                request.session_id,
                block.groundedness,
                block.relevance,
                block.confidence,
                block.flagged,
            )
            return block

        except LLMError as exc:
            logger.error(
                "[%s] Eval LLM error | user=%s session=%s error=%s",
                request.request_id,
                request.user_id,
                request.session_id,
                exc,
            )
            return _safe_default_eval(
                f"Evaluation service error: {exc}. Response flagged for safety."
            )
        except (TypeError, ValueError) as exc:
            logger.error(
                "[%s] Eval parse error | user=%s session=%s error=%s",
                request.request_id,
                request.user_id,
                request.session_id,
                exc,
            )
            return _safe_default_eval(
                f"Evaluation parse error: {exc}. Response flagged for safety."
            )
        except Exception as exc:
            logger.exception(
                "[%s] Eval failed | user=%s session=%s",
                request.request_id,
                request.user_id,
                request.session_id,
            )
            return _safe_default_eval(
                f"Evaluation service error: {exc}. Response flagged for safety."
            )

    @staticmethod
    def _parse_eval_data(data: dict) -> EvalBlock:
        groundedness = float(data.get("groundedness", 0.5))
        relevance = float(data.get("relevance", 0.5))
        confidence = float(data.get("confidence", 0.5))
        threshold = settings.EVAL_FLAG_THRESHOLD

        flagged = (
            bool(data.get("flagged", False))
            or confidence < threshold
            or groundedness < threshold
            or relevance < threshold
        )
        reasoning = str(data.get("reasoning", "Evaluation completed."))

        return EvalBlock(
            groundedness=round(_clamp(groundedness), 4),
            relevance=round(_clamp(relevance), 4),
            confidence=round(_clamp(confidence), 4),
            flagged=flagged,
            reasoning=reasoning,
        )


def _clamp(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def _safe_default_eval(reasoning: str) -> EvalBlock:
    return EvalBlock(
        groundedness=0.5,
        relevance=0.5,
        confidence=0.4,
        flagged=True,
        reasoning=reasoning,
    )


# ── Module-level convenience (backward compatible) ────────────────────────────


def evaluate_response(
    user_message: str,
    assistant_response: str,
    catalog_context: str,
    memory_context: str,
    llm: LLMService | None = None,
    *,
    user_id: str = "",
    session_id: str = "",
    request_id: str = "",
) -> EvalBlock:
    """Evaluate a response using a default EvalService instance."""
    service = EvalService(llm=llm)
    return service.evaluate(
        EvalRequest(
            user_message=user_message,
            assistant_response=assistant_response,
            catalog_context=catalog_context,
            memory_context=memory_context,
            user_id=user_id,
            session_id=session_id,
            request_id=request_id or "legacy",
        )
    )
