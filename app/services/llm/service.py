"""High-level LLM service used by agents and evaluators."""
from __future__ import annotations

import logging
from typing import Any

from app.services.llm.base import BaseLLMProvider
from app.services.llm.exceptions import LLMError

logger = logging.getLogger(__name__)


class LLMService:
    """
    Provider-agnostic facade for text and JSON completions.

    Business code depends on this class — not on Groq, OpenAI, or Anthropic SDKs.
    """

    def __init__(self, provider: BaseLLMProvider) -> None:
        self._provider = provider

    @property
    def provider_name(self) -> str:
        return self._provider.provider_name

    @property
    def model_name(self) -> str:
        return self._provider.model_name

    def complete(
        self,
        *,
        system: str,
        user_message: str,
        max_tokens: int,
        temperature: float = 0.3,
    ) -> str:
        logger.info(
            "LLM complete | provider=%s model=%s max_tokens=%d",
            self.provider_name,
            self.model_name,
            max_tokens,
        )
        try:
            return self._provider.complete(
                system=system,
                user_message=user_message,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except LLMError:
            raise
        except Exception as exc:
            logger.exception("Unexpected LLM complete failure")
            raise LLMError(f"LLM completion failed: {exc}") from exc

    def complete_json(
        self,
        *,
        system: str,
        user_message: str,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        logger.info(
            "LLM complete_json | provider=%s model=%s max_tokens=%d",
            self.provider_name,
            self.model_name,
            max_tokens,
        )
        try:
            return self._provider.complete_json(
                system=system,
                user_message=user_message,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except LLMError:
            raise
        except Exception as exc:
            logger.exception("Unexpected LLM JSON completion failure")
            raise LLMError(f"LLM JSON completion failed: {exc}") from exc
