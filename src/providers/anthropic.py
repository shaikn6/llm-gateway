"""Anthropic (Claude) provider implementation."""

from __future__ import annotations

import os
import time
import uuid
from typing import AsyncIterator, List, Optional

import anthropic

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

# Models supported and their context windows
ANTHROPIC_MODELS: dict[str, int] = {
    "claude-3-5-sonnet-20241022": 200_000,
    "claude-3-5-haiku-20241022": 200_000,
    "claude-3-opus-20240229": 200_000,
    "claude-3-sonnet-20240229": 200_000,
    "claude-3-haiku-20240307": 200_000,
}


def _convert_messages(
    request: ChatCompletionRequest,
) -> tuple[Optional[str], list[dict]]:
    """Extract system prompt and convert messages to Anthropic format."""
    system_prompt: Optional[str] = None
    messages: list[dict] = []

    for msg in request.messages:
        role = msg.role.value
        content = msg.content or ""

        if role == "system":
            system_prompt = content if isinstance(content, str) else str(content)
            continue

        if role == "tool":
            # Map tool results back to user role
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.tool_call_id or "",
                            "content": content
                            if isinstance(content, str)
                            else str(content),
                        }
                    ],
                }
            )
            continue

        if role == "assistant" and msg.tool_calls:
            tool_use_blocks = []
            for tc in msg.tool_calls:
                tool_use_blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": tc.get("function", {}).get("name", ""),
                        "input": tc.get("function", {}).get("arguments", {}),
                    }
                )
            messages.append({"role": "assistant", "content": tool_use_blocks})
            continue

        # Ensure alternating user/assistant (Anthropic requirement)
        anthropic_role = "user" if role == "user" else "assistant"
        if isinstance(content, list):
            messages.append({"role": anthropic_role, "content": content})
        else:
            messages.append({"role": anthropic_role, "content": str(content)})

    return system_prompt, messages


class AnthropicProvider(LLMProvider):
    """Claude provider via the official anthropic SDK."""

    name = "anthropic"

    def __init__(self, api_key: Optional[str] = None, timeout: float = 60.0):
        super().__init__(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"), timeout=timeout
        )
        if not self.api_key:
            raise ProviderAuthError("ANTHROPIC_API_KEY is not set", provider=self.name)
        self._client = anthropic.AsyncAnthropic(api_key=self.api_key, timeout=timeout)

    def _build_kwargs(self, request: ChatCompletionRequest) -> dict:
        system_prompt, messages = _convert_messages(request)

        kwargs: dict = {
            "model": request.model,
            "messages": messages,
            "max_tokens": request.effective_max_tokens() or 4096,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p
        if request.stop:
            kwargs["stop_sequences"] = (
                [request.stop] if isinstance(request.stop, str) else request.stop
            )
        return kwargs

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        kwargs = self._build_kwargs(request)
        try:
            resp = await self._client.messages.create(**kwargs)
        except anthropic.AuthenticationError as exc:
            raise ProviderAuthError(str(exc), provider=self.name) from exc
        except anthropic.RateLimitError as exc:
            raise ProviderRateLimitError(str(exc), provider=self.name) from exc
        except anthropic.APITimeoutError as exc:
            raise ProviderTimeoutError(str(exc), provider=self.name) from exc
        except anthropic.APIError as exc:
            raise ProviderError(
                str(exc),
                status_code=getattr(exc, "status_code", 500),
                provider=self.name,
            ) from exc

        content_text = ""
        for block in resp.content:
            if hasattr(block, "text"):
                content_text += block.text

        usage = UsageInfo(
            prompt_tokens=resp.usage.input_tokens,
            completion_tokens=resp.usage.output_tokens,
            total_tokens=resp.usage.input_tokens + resp.usage.output_tokens,
        )

        return ChatCompletionResponse(
            id=f"chatcmpl-{resp.id}",
            model=resp.model,
            choices=[
                Choice(
                    index=0,
                    message=ChoiceMessage(role="assistant", content=content_text),
                    finish_reason=resp.stop_reason or "stop",
                )
            ],
            usage=usage,
        )

    async def stream(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[ChatCompletionChunk]:
        kwargs = self._build_kwargs(request)
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        model = request.model

        try:
            async with self._client.messages.stream(**kwargs) as stream:
                async for event in stream:
                    if hasattr(event, "type"):
                        if event.type == "content_block_delta":
                            delta_text = getattr(event.delta, "text", "")
                            yield ChatCompletionChunk(
                                id=completion_id,
                                model=model,
                                choices=[
                                    StreamChoice(
                                        index=0,
                                        delta=ChoiceDelta(
                                            role="assistant", content=delta_text
                                        ),
                                        finish_reason=None,
                                    )
                                ],
                            )
                        elif event.type == "message_stop":
                            yield ChatCompletionChunk(
                                id=completion_id,
                                model=model,
                                choices=[
                                    StreamChoice(
                                        index=0,
                                        delta=ChoiceDelta(),
                                        finish_reason="stop",
                                    )
                                ],
                            )
        except anthropic.AuthenticationError as exc:
            raise ProviderAuthError(str(exc), provider=self.name) from exc
        except anthropic.RateLimitError as exc:
            raise ProviderRateLimitError(str(exc), provider=self.name) from exc
        except anthropic.APITimeoutError as exc:
            raise ProviderTimeoutError(str(exc), provider=self.name) from exc
        except anthropic.APIError as exc:
            raise ProviderError(
                str(exc),
                status_code=getattr(exc, "status_code", 500),
                provider=self.name,
            ) from exc

    async def list_models(self) -> List[ModelObject]:
        return [
            ModelObject(id=model_id, owned_by="anthropic")
            for model_id in ANTHROPIC_MODELS
        ]

    async def health_check(self) -> HealthStatus:
        start = time.monotonic()
        try:
            # Minimal API call to verify connectivity
            await self._client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1,
                messages=[{"role": "user", "content": "ping"}],
            )
            latency_ms = (time.monotonic() - start) * 1000
            return HealthStatus(provider=self.name, healthy=True, latency_ms=latency_ms)
        except Exception as exc:
            latency_ms = (time.monotonic() - start) * 1000
            return HealthStatus(
                provider=self.name, healthy=False, latency_ms=latency_ms, error=str(exc)
            )
