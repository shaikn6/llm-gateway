"""OpenAI-compatible request/response schemas and internal models."""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# OpenAI-compatible request models
# ---------------------------------------------------------------------------


class MessageRole(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"
    tool = "tool"
    function = "function"


class ChatMessage(BaseModel):
    role: MessageRole
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class ResponseFormat(BaseModel):
    type: Literal["text", "json_object"] = "text"


class StreamOptions(BaseModel):
    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float | None = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float | None = Field(default=1.0, ge=0.0, le=1.0)
    n: int | None = Field(default=1, ge=1, le=8)
    stream: bool | None = False
    stream_options: StreamOptions | None = None
    stop: str | list[str] | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    presence_penalty: float | None = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: float | None = Field(default=0.0, ge=-2.0, le=2.0)
    logit_bias: dict[str, float] | None = None
    user: str | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    response_format: ResponseFormat | None = None
    seed: int | None = None

    # Gateway-specific extensions (ignored by OpenAI clients)
    x_budget_usd: float | None = Field(default=None, alias="x-budget-usd")
    x_provider: str | None = Field(default=None, alias="x-provider")
    x_latency_target_ms: int | None = Field(
        default=None, alias="x-latency-target-ms"
    )

    class Config:
        populate_by_name = True

    @field_validator("messages")
    @classmethod
    def messages_not_empty(cls, v: list[ChatMessage]) -> list[ChatMessage]:
        if not v:
            raise ValueError("messages must not be empty")
        return v

    def effective_max_tokens(self) -> int | None:
        return self.max_completion_tokens or self.max_tokens


# ---------------------------------------------------------------------------
# OpenAI-compatible response models
# ---------------------------------------------------------------------------


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChoiceMessage(BaseModel):
    role: str = "assistant"
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class Choice(BaseModel):
    index: int
    message: ChoiceMessage
    finish_reason: str | None = "stop"
    logprobs: Any | None = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[Choice]
    usage: UsageInfo = Field(default_factory=UsageInfo)
    system_fingerprint: str | None = None

    # Gateway metadata (extra fields)
    x_gateway_provider: str | None = Field(default=None, alias="x-gateway-provider")
    x_gateway_cache: str | None = Field(default=None, alias="x-gateway-cache")
    x_gateway_cost_usd: float | None = Field(
        default=None, alias="x-gateway-cost-usd"
    )
    x_gateway_latency_ms: int | None = Field(
        default=None, alias="x-gateway-latency-ms"
    )

    class Config:
        populate_by_name = True


# Streaming delta models


class ChoiceDelta(BaseModel):
    role: str | None = None
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class StreamChoice(BaseModel):
    index: int
    delta: ChoiceDelta
    finish_reason: str | None = None
    logprobs: Any | None = None


class ChatCompletionChunk(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[StreamChoice]
    usage: UsageInfo | None = None
    system_fingerprint: str | None = None


# ---------------------------------------------------------------------------
# Models list response
# ---------------------------------------------------------------------------


class ModelObject(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "llm-gateway"


class ModelsListResponse(BaseModel):
    object: str = "list"
    data: list[ModelObject]


# ---------------------------------------------------------------------------
# Internal gateway models
# ---------------------------------------------------------------------------


class ProviderName(str, Enum):
    anthropic = "anthropic"
    openai = "openai"
    ollama = "ollama"


class RoutingDecision(BaseModel):
    provider: ProviderName
    model: str
    reason: str
    original_model: str
    fallback_chain: list[tuple] = Field(default_factory=list)


class CacheResult(BaseModel):
    hit: bool
    response: ChatCompletionResponse | None = None
    similarity: float = 0.0
    cache_key: str | None = None


class CostRecord(BaseModel):
    request_id: str
    api_key_hash: str
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    latency_ms: int
    cache_hit: bool
    timestamp: float = Field(default_factory=time.time)


class RateLimitInfo(BaseModel):
    allowed: bool
    remaining_requests: int
    remaining_tokens: int
    reset_at: float
    retry_after: float | None = None


class HealthStatus(BaseModel):
    provider: str
    healthy: bool
    latency_ms: float | None = None
    error: str | None = None
    last_checked: float = Field(default_factory=time.time)


class GatewayStats(BaseModel):
    total_requests: int
    cache_hit_rate: float
    total_cost_usd: float
    requests_by_provider: dict[str, int]
    avg_latency_ms: float
    error_rate: float
    uptime_seconds: float


class ErrorResponse(BaseModel):
    error: dict[str, Any]

    @classmethod
    def create(
        cls,
        message: str,
        type_: str = "invalid_request_error",
        code: str | None = None,
    ) -> ErrorResponse:
        return cls(
            error={"message": message, "type": type_, "code": code, "param": None}
        )
