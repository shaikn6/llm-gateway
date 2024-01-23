"""Abstract base class for all LLM providers."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import AsyncIterator, List, Optional

from src.models.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChunk,
    HealthStatus,
    ModelObject,
)


class ProviderError(Exception):
    """Base exception for provider errors."""

    def __init__(self, message: str, status_code: int = 500, provider: str = "unknown"):
        super().__init__(message)
        self.status_code = status_code
        self.provider = provider


class ProviderRateLimitError(ProviderError):
    """Raised when the upstream provider returns a rate limit error."""

    def __init__(
        self,
        message: str,
        retry_after: Optional[float] = None,
        provider: str = "unknown",
    ):
        super().__init__(message, status_code=429, provider=provider)
        self.retry_after = retry_after


class ProviderAuthError(ProviderError):
    """Raised when authentication with the upstream provider fails."""

    def __init__(self, message: str, provider: str = "unknown"):
        super().__init__(message, status_code=401, provider=provider)


class ProviderTimeoutError(ProviderError):
    """Raised when the upstream provider times out."""

    def __init__(self, message: str, provider: str = "unknown"):
        super().__init__(message, status_code=504, provider=provider)


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    Every provider must implement:
    - complete(): non-streaming completion
    - stream(): async generator yielding SSE chunks
    - list_models(): enumerate available models
    - health_check(): probe provider liveness
    """

    name: str = "base"

    def __init__(self, api_key: Optional[str] = None, timeout: float = 60.0):
        self.api_key = api_key
        self.timeout = timeout
        self._healthy: bool = True
        self._last_health_check: float = 0.0
        self._health_check_interval: float = 30.0  # seconds

    @abstractmethod
    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Send a non-streaming chat completion request."""
        ...

    @abstractmethod
    async def stream(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Send a streaming chat completion request, yielding chunks."""
        ...

    @abstractmethod
    async def list_models(self) -> List[ModelObject]:
        """Return the list of models available from this provider."""
        ...

    @abstractmethod
    async def health_check(self) -> HealthStatus:
        """Probe the provider and return a health status."""
        ...

    async def is_healthy(self) -> bool:
        """Return cached health or re-probe if the interval has elapsed."""
        now = time.monotonic()
        if now - self._last_health_check > self._health_check_interval:
            status = await self.health_check()
            self._healthy = status.healthy
            self._last_health_check = now
        return self._healthy

    def _count_tokens_estimate(self, text: str) -> int:
        """Rough token estimate: ~4 chars per token (OpenAI heuristic)."""
        return max(1, len(text) // 4)
