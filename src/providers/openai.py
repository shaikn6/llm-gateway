"""OpenAI provider — full async implementation conforming to LLMProvider."""

from __future__ import annotations

import os
import time
import uuid
from typing import AsyncIterator, List, Optional

import openai as openai_lib

from src.models.schemas import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChoiceDelta,
    ChoiceMessage,
    HealthStatus,
    ModelObject,
    StreamChoice,
    UsageInfo,
)
from src.providers.base import (
    LLMProvider,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)

OPENAI_MODELS: dict[str, int] = {
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-3.5-turbo": 16_385,
    "o1-preview": 128_000,
    "o1-mini": 128_000,
}


class OpenAIAsyncProvider(LLMProvider):
    """OpenAI provider via the official openai SDK."""

    name = "openai"

    def __init__(self, api_key: Optional[str] = None, timeout: float = 60.0):
        super().__init__(api_key=api_key or os.environ.get("OPENAI_API_KEY"), timeout=timeout)
        if not self.api_key:
            raise ProviderAuthError("OPENAI_API_KEY is not set", provider=self.name)
        self._client = openai_lib.AsyncOpenAI(api_key=self.api_key, timeout=timeout)

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        messages = [{"role": m.role.value, "content": m.content or ""} for m in request.messages]
        kwargs: dict = {
            "model": request.model,
            "messages": messages,
        }
        if request.effective_max_tokens():
            kwargs["max_tokens"] = request.effective_max_tokens()
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature

        try:
            resp = await self._client.chat.completions.create(**kwargs)
        except openai_lib.AuthenticationError as exc:
            raise ProviderAuthError(str(exc), provider=self.name) from exc
        except openai_lib.RateLimitError as exc:
            raise ProviderRateLimitError(str(exc), provider=self.name) from exc
        except openai_lib.APITimeoutError as exc:
            raise ProviderTimeoutError(str(exc), provider=self.name) from exc
        except openai_lib.APIError as exc:
            raise ProviderError(
                str(exc),
                status_code=getattr(exc, "status_code", 500),
                provider=self.name,
            ) from exc

        content = resp.choices[0].message.content or ""
        usage = UsageInfo(
            prompt_tokens=resp.usage.prompt_tokens,
            completion_tokens=resp.usage.completion_tokens,
            total_tokens=resp.usage.total_tokens,
        )
        return ChatCompletionResponse(
            id=resp.id,
            model=resp.model,
            choices=[
                Choice(
                    index=0,
                    message=ChoiceMessage(role="assistant", content=content),
                    finish_reason=resp.choices[0].finish_reason or "stop",
                )
            ],
            usage=usage,
        )

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[ChatCompletionChunk]:
        messages = [{"role": m.role.value, "content": m.content or ""} for m in request.messages]
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        try:
            async with await self._client.chat.completions.create(
                model=request.model,
                messages=messages,
                stream=True,
            ) as stream:
                async for chunk in stream:
                    delta_content = chunk.choices[0].delta.content if chunk.choices else None
                    finish = chunk.choices[0].finish_reason if chunk.choices else None
                    yield ChatCompletionChunk(
                        id=completion_id,
                        model=request.model,
                        choices=[
                            StreamChoice(
                                index=0,
                                delta=ChoiceDelta(role="assistant", content=delta_content),
                                finish_reason=finish,
                            )
                        ],
                    )
        except openai_lib.APIError as exc:
            raise ProviderError(str(exc), provider=self.name) from exc

    async def list_models(self) -> List[ModelObject]:
        return [ModelObject(id=model_id, owned_by="openai") for model_id in OPENAI_MODELS]

    async def health_check(self) -> HealthStatus:
        start = time.monotonic()
        try:
            await self._client.models.list()
            latency_ms = (time.monotonic() - start) * 1000
            return HealthStatus(provider=self.name, healthy=True, latency_ms=latency_ms)
        except Exception as exc:
            latency_ms = (time.monotonic() - start) * 1000
            return HealthStatus(
                provider=self.name, healthy=False, latency_ms=latency_ms, error=str(exc)
            )
