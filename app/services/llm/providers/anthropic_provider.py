"""Anthropic provider — optional backend for provider swapping."""
from __future__ import annotations

import logging
from typing import Any

from app.services.llm.base import BaseLLMProvider
from app.services.llm.exceptions import LLMConfigurationError, LLMProviderError, LLMTimeoutError
from app.services.llm.json_utils import parse_json_object

logger = logging.getLogger(__name__)

_TIMEOUT_ERRORS = frozenset({"APITimeoutError", "APITimeError", "ReadTimeout", "ConnectTimeout"})


class AnthropicProvider(BaseLLMProvider):
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout: float = 60.0,
    ) -> None:
        if not api_key:
            raise LLMConfigurationError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic")

        try:
            import anthropic
        except ImportError as exc:
            raise LLMConfigurationError(
                "anthropic package is required for LLM_PROVIDER=anthropic. "
                "Install with: pip install anthropic"
            ) from exc

        self._anthropic = anthropic
        self._model = model
        self._client = anthropic.Anthropic(api_key=api_key, timeout=timeout, max_retries=2)

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def model_name(self) -> str:
        return self._model

    def _handle_error(self, exc: Exception, action: str) -> None:
        if type(exc).__name__ in _TIMEOUT_ERRORS:
            raise LLMTimeoutError(f"Anthropic {action} timed out") from exc
        raise LLMProviderError(f"Anthropic {action} error: {exc}") from exc

    def complete(
        self,
        *,
        system: str,
        user_message: str,
        max_tokens: int,
        temperature: float = 0.3,
    ) -> str:
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": user_message}],
            )
        except Exception as exc:
            self._handle_error(exc, "completion")

        if not response.content:
            raise LLMProviderError("Anthropic returned empty completion")
        return response.content[0].text.strip()

    def complete_json(
        self,
        *,
        system: str,
        user_message: str,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        json_system = (
            f"{system.strip()}\n\n"
            "Respond with ONLY a single valid JSON object. No markdown fences."
        )
        raw = self.complete(
            system=json_system,
            user_message=user_message,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return parse_json_object(raw)
