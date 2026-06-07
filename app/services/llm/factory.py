"""Factory — instantiate the configured LLM provider."""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from app.config import Settings, get_settings
from app.services.llm.base import BaseLLMProvider
from app.services.llm.exceptions import LLMConfigurationError
from app.services.llm.providers.anthropic_provider import AnthropicProvider
from app.services.llm.providers.openai_compatible import OpenAICompatibleProvider
from app.services.llm.service import LLMService

ProviderName = Literal["groq", "openai", "anthropic"]


def create_llm_provider(settings: Settings | None = None) -> BaseLLMProvider:
    """Build the LLM provider from application settings."""
    cfg = settings or get_settings()
    provider = cfg.LLM_PROVIDER.lower()

    if provider == "groq":
        if not cfg.GROQ_API_KEY:
            raise LLMConfigurationError("GROQ_API_KEY is required when LLM_PROVIDER=groq")
        return OpenAICompatibleProvider(
            provider_name="groq",
            api_key=cfg.GROQ_API_KEY,
            model=cfg.resolved_model_name(),
            base_url=cfg.GROQ_BASE_URL,
            timeout=cfg.LLM_TIMEOUT_SECONDS,
        )

    if provider == "openai":
        if not cfg.OPENAI_API_KEY:
            raise LLMConfigurationError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        return OpenAICompatibleProvider(
            provider_name="openai",
            api_key=cfg.OPENAI_API_KEY,
            model=cfg.resolved_model_name(),
            base_url=cfg.OPENAI_BASE_URL or None,
            timeout=cfg.LLM_TIMEOUT_SECONDS,
        )

    if provider == "anthropic":
        return AnthropicProvider(
            api_key=cfg.ANTHROPIC_API_KEY,
            model=cfg.resolved_model_name(),
            timeout=cfg.LLM_TIMEOUT_SECONDS,
        )

    raise LLMConfigurationError(
        f"Unsupported LLM_PROVIDER={cfg.LLM_PROVIDER!r}. "
        "Use one of: groq, openai, anthropic."
    )


@lru_cache
def get_llm_service() -> LLMService:
    """Return a cached LLMService bound to current settings."""
    return LLMService(create_llm_provider())
