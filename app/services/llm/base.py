"""Abstract LLM provider interface — swap backends without touching business logic."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseLLMProvider(ABC):
    """Contract every LLM backend (Groq, OpenAI, Anthropic) must implement."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...

    @abstractmethod
    def complete(
        self,
        *,
        system: str,
        user_message: str,
        max_tokens: int,
        temperature: float = 0.3,
    ) -> str:
        """Return plain-text completion."""
        ...

    @abstractmethod
    def complete_json(
        self,
        *,
        system: str,
        user_message: str,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """Return a parsed JSON object from the model."""
        ...
