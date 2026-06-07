from functools import lru_cache
from typing import Literal  # noqa: TC003 — used by pydantic field annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",  # ignore legacy vars e.g. ANTHROPIC_API_KEY, AGENT_MODEL
    )

    # App
    APP_NAME: str = "Sales Assistant Agent"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # LLM provider — swap with LLM_PROVIDER (groq | openai | anthropic)
    LLM_PROVIDER: Literal["groq", "openai", "anthropic"] = "groq"
    MODEL_NAME: str = "llama-3.3-70b-versatile"
    LLM_TIMEOUT_SECONDS: float = 60.0

    # Provider API keys (only the active provider's key is required)
    GROQ_API_KEY: str = ""
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = ""
    ANTHROPIC_API_KEY: str = ""

    # Database
    DATABASE_URL: str = "sqlite:///./sales_agent.db"
    MEMORY_BACKEND: Literal["sqlite", "postgres", "mem0"] = "sqlite"

    # Agent
    AGENT_MAX_TOKENS: int = 2048
    MEMORY_MAX_MESSAGES: int = 20
    MEMORY_SUMMARY_THRESHOLD: int = 30

    # Eval thresholds
    EVAL_FLAG_THRESHOLD: float = 0.60

    # Catalog
    CATALOG_PATH: str = "app/catalog.json"

    # Legacy env var — superseded by MODEL_NAME; accepted so old .env files still load
    AGENT_MODEL: str = ""

    def resolved_model_name(self) -> str:
        """MODEL_NAME with fallback for legacy AGENT_MODEL env var."""
        if self.MODEL_NAME:
            return self.MODEL_NAME
        if self.AGENT_MODEL:
            return self.AGENT_MODEL
        return "llama-3.3-70b-versatile"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
