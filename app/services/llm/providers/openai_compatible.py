"""
OpenAI-compatible provider — used by Groq and OpenAI.

Groq exposes an OpenAI-compatible API at https://api.groq.com/openai/v1.
"""
from __future__ import annotations

import logging
from typing import Any

from openai import APIConnectionError, APITimeoutError, APIStatusError, OpenAI

from app.services.llm.base import BaseLLMProvider
from app.services.llm.exceptions import LLMConfigurationError, LLMProviderError, LLMTimeoutError
from app.services.llm.json_utils import parse_json_object

logger = logging.getLogger(__name__)


class OpenAICompatibleProvider(BaseLLMProvider):
    def __init__(
        self,
        *,
        provider_name: str,
        api_key: str,
        model: str,
        base_url: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        if not api_key:
            raise LLMConfigurationError(f"{provider_name} API key is not configured")

        self._provider_name = provider_name
        self._model = model
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=2,
        )

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def model_name(self) -> str:
        return self._model

    def complete(
        self,
        *,
        system: str,
        user_message: str,
        max_tokens: int,
        temperature: float = 0.3,
    ) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except APITimeoutError as exc:
            raise LLMTimeoutError(f"{self._provider_name} request timed out") from exc
        except (APIStatusError, APIConnectionError) as exc:
            raise LLMProviderError(f"{self._provider_name} API error: {exc}") from exc

        content = response.choices[0].message.content
        if not content:
            raise LLMProviderError(f"{self._provider_name} returned empty completion")
        return content.strip()

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
            "You MUST respond with a single valid JSON object only. "
            "No markdown fences, no commentary outside the JSON."
        )

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": json_system},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                response_format={"type": "json_object"},
            )
        except APITimeoutError as exc:
            raise LLMTimeoutError(f"{self._provider_name} JSON request timed out") from exc
        except (APIStatusError, APIConnectionError) as exc:
            raise LLMProviderError(f"{self._provider_name} JSON API error: {exc}") from exc

        raw = response.choices[0].message.content or ""
        if not raw.strip():
            raise LLMProviderError(f"{self._provider_name} returned empty JSON completion")

        return parse_json_object(raw)
