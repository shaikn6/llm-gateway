"""OpenAI-compatible request/response schemas and internal models."""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

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
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class ResponseFormat(BaseModel):
    type: Literal["text", "json_object"] = "text"


class StreamOptions(BaseModel):
    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = Field(default=1.0, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    n: Optional[int] = Field(default=1, ge=1, le=8)
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptions] = None
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = Field(default=None, ge=1)
    max_completion_tokens: Optional[int] = Field(default=None, ge=1)
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    response_format: Optional[ResponseFormat] = None
    seed: Optional[int] = None

    # Gateway-specific extensions (ignored by OpenAI clients)
    x_budget_usd: Optional[float] = Field(default=None, alias="x-budget-usd")
    x_provider: Optional[str] = Field(default=None, alias="x-provider")
    x_latency_target_ms: Optional[int] = Field(
        default=None, alias="x-latency-target-ms"
    )

    class Config:
        populate_by_name = True

    @field_validator("messages")
    @classmethod
    def messages_not_empty(cls, v: List[ChatMessage]) -> List[ChatMessage]:
        if not v:
            raise ValueError("messages must not be empty")
        return v

    def effective_max_tokens(self) -> Optional[int]:
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
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class Choice(BaseModel):
    index: int
    message: ChoiceMessage
    finish_reason: Optional[str] = "stop"
    logprobs: Optional[Any] = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[Choice]
    usage: UsageInfo = Field(default_factory=UsageInfo)
    system_fingerprint: Optional[str] = None

    # Gateway metadata (extra fields)
    x_gateway_provider: Optional[str] = Field(default=None, alias="x-gateway-provider")
    x_gateway_cache: Optional[str] = Field(default=None, alias="x-gateway-cache")
    x_gateway_cost_usd: Optional[float] = Field(
        default=None, alias="x-gateway-cost-usd"
    )
    x_gateway_latency_ms: Optional[int] = Field(
        default=None, alias="x-gateway-latency-ms"
    )

    class Config:
        populate_by_name = True


# Streaming delta models


class ChoiceDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class StreamChoice(BaseModel):
    index: int
    delta: ChoiceDelta
    finish_reason: Optional[str] = None
    logprobs: Optional[Any] = None


class ChatCompletionChunk(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[StreamChoice]
    usage: Optional[UsageInfo] = None
    system_fingerprint: Optional[str] = None


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
    data: List[ModelObject]


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
    fallback_chain: List[tuple] = Field(default_factory=list)


class CacheResult(BaseModel):
    hit: bool
    response: Optional[ChatCompletionResponse] = None
    similarity: float = 0.0
    cache_key: Optional[str] = None


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
    retry_after: Optional[float] = None


class HealthStatus(BaseModel):
    provider: str
    healthy: bool
    latency_ms: Optional[float] = None
    error: Optional[str] = None
    last_checked: float = Field(default_factory=time.time)


class GatewayStats(BaseModel):
    total_requests: int
    cache_hit_rate: float
    total_cost_usd: float
    requests_by_provider: Dict[str, int]
    avg_latency_ms: float
    error_rate: float
    uptime_seconds: float


class ErrorResponse(BaseModel):
    error: Dict[str, Any]

    @classmethod
    def create(
        cls,
        message: str,
        type_: str = "invalid_request_error",
        code: Optional[str] = None,
    ) -> "ErrorResponse":
        return cls(
            error={"message": message, "type": type_, "code": code, "param": None}
        )
