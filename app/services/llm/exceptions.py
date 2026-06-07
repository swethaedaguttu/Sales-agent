"""LLM provider exceptions — provider-agnostic error types."""


class LLMError(Exception):
    """Base exception for all LLM service failures."""


class LLMTimeoutError(LLMError):
    """Raised when an LLM request exceeds the configured timeout."""


class LLMProviderError(LLMError):
    """Raised when the upstream provider returns an error response."""


class LLMConfigurationError(LLMError):
    """Raised when the LLM provider is misconfigured."""


class LLMJSONParseError(LLMError):
    """Raised when structured JSON output cannot be parsed."""
