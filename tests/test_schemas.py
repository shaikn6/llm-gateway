"""Unit tests for src/models/schemas.py."""

from __future__ import annotations

import time

import pytest
from pydantic import ValidationError

from src.models.schemas import (
    CacheResult,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    ChoiceDelta,
    ChoiceMessage,
    CostRecord,
    ErrorResponse,
    GatewayStats,
    HealthStatus,
    MessageRole,
    ModelObject,
    ModelsListResponse,
    ProviderName,
    RateLimitInfo,
    ResponseFormat,
    RoutingDecision,
    StreamChoice,
    StreamOptions,
    UsageInfo,
)


class TestMessageRole:
    def test_all_roles_exist(self):
        roles = {r.value for r in MessageRole}
        assert roles == {"system", "user", "assistant", "tool", "function"}

    def test_user_role(self):
        assert MessageRole.user == "user"


class TestChatMessage:
    def test_valid_user_message(self):
        msg = ChatMessage(role=MessageRole.user, content="Hello")
        assert msg.role == MessageRole.user
        assert msg.content == "Hello"

    def test_optional_fields_default_none(self):
        msg = ChatMessage(role=MessageRole.user, content="Hi")
        assert msg.name is None
        assert msg.tool_call_id is None
        assert msg.tool_calls is None

    def test_assistant_with_tool_calls(self):
        tool_calls = [{"id": "tc1", "function": {"name": "search", "arguments": "{}"}}]
        msg = ChatMessage(role=MessageRole.assistant, content=None, tool_calls=tool_calls)
        assert msg.tool_calls == tool_calls

    def test_list_content(self):
        content = [{"type": "text", "text": "Hello"}]
        msg = ChatMessage(role=MessageRole.user, content=content)
        assert isinstance(msg.content, list)


class TestChatCompletionRequest:
    def test_minimal_valid_request(self):
        req = ChatCompletionRequest(
            model="claude-haiku-4-5",
            messages=[ChatMessage(role=MessageRole.user, content="Hi")],
        )
        assert req.model == "claude-haiku-4-5"
        assert len(req.messages) == 1

    def test_empty_messages_raises(self):
        with pytest.raises(ValidationError, match="messages must not be empty"):
            ChatCompletionRequest(model="claude-haiku-4-5", messages=[])

    def test_default_temperature(self):
        req = ChatCompletionRequest(
            model="gpt-4o",
            messages=[ChatMessage(role=MessageRole.user, content="Hi")],
        )
        assert req.temperature == 1.0

    def test_temperature_out_of_range_raises(self):
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="gpt-4o",
                messages=[ChatMessage(role=MessageRole.user, content="Hi")],
                temperature=3.0,
            )

    def test_stream_defaults_false(self):
        req = ChatCompletionRequest(
            model="gpt-4o",
            messages=[ChatMessage(role=MessageRole.user, content="Hi")],
        )
        assert req.stream is False

    def test_effective_max_tokens_uses_max_completion_tokens(self):
        req = ChatCompletionRequest(
            model="gpt-4o",
            messages=[ChatMessage(role=MessageRole.user, content="Hi")],
            max_completion_tokens=512,
            max_tokens=256,
        )
        assert req.effective_max_tokens() == 512

    def test_effective_max_tokens_falls_back_to_max_tokens(self):
        req = ChatCompletionRequest(
            model="gpt-4o",
            messages=[ChatMessage(role=MessageRole.user, content="Hi")],
            max_tokens=300,
        )
        assert req.effective_max_tokens() == 300

    def test_effective_max_tokens_returns_none_when_unset(self):
        req = ChatCompletionRequest(
            model="gpt-4o",
            messages=[ChatMessage(role=MessageRole.user, content="Hi")],
        )
        assert req.effective_max_tokens() is None

    def test_top_p_out_of_range_raises(self):
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="gpt-4o",
                messages=[ChatMessage(role=MessageRole.user, content="Hi")],
                top_p=1.5,
            )

    def test_max_tokens_negative_raises(self):
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="gpt-4o",
                messages=[ChatMessage(role=MessageRole.user, content="Hi")],
                max_tokens=-1,
            )

    def test_response_format_json_object(self):
        req = ChatCompletionRequest(
            model="gpt-4o",
            messages=[ChatMessage(role=MessageRole.user, content="Hi")],
            response_format=ResponseFormat(type="json_object"),
        )
        assert req.response_format.type == "json_object"

    def test_stop_string(self):
        req = ChatCompletionRequest(
            model="gpt-4o",
            messages=[ChatMessage(role=MessageRole.user, content="Hi")],
            stop="\n",
        )
        assert req.stop == "\n"

    def test_stop_list(self):
        req = ChatCompletionRequest(
            model="gpt-4o",
            messages=[ChatMessage(role=MessageRole.user, content="Hi")],
            stop=["\n", "END"],
        )
        assert req.stop == ["\n", "END"]

    def test_x_provider_alias(self):
        req = ChatCompletionRequest(
            model="gpt-4o",
            messages=[ChatMessage(role=MessageRole.user, content="Hi")],
            **{"x-provider": "anthropic"},
        )
        assert req.x_provider == "anthropic"


class TestUsageInfo:
    def test_defaults_are_zero(self):
        u = UsageInfo()
        assert u.prompt_tokens == 0
        assert u.completion_tokens == 0
        assert u.total_tokens == 0

    def test_explicit_values(self):
        u = UsageInfo(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        assert u.total_tokens == 30


class TestChatCompletionResponse:
    def _make_response(self, **kwargs):
        defaults = dict(
            model="claude-haiku-4-5",
            choices=[
                Choice(
                    index=0,
                    message=ChoiceMessage(role="assistant", content="Hi"),
                    finish_reason="stop",
                )
            ],
        )
        defaults.update(kwargs)
        return ChatCompletionResponse(**defaults)

    def test_id_starts_with_chatcmpl(self):
        resp = self._make_response()
        assert resp.id.startswith("chatcmpl-")

    def test_object_is_chat_completion(self):
        resp = self._make_response()
        assert resp.object == "chat.completion"

    def test_created_is_recent_timestamp(self):
        before = int(time.time()) - 2
        resp = self._make_response()
        assert resp.created >= before

    def test_gateway_metadata_aliases(self):
        resp = self._make_response(
            **{
                "x-gateway-provider": "openai",
                "x-gateway-cache": "hit",
                "x-gateway-cost-usd": 0.001,
                "x-gateway-latency-ms": 250,
            }
        )
        assert resp.x_gateway_provider == "openai"
        assert resp.x_gateway_cache == "hit"
        assert resp.x_gateway_cost_usd == pytest.approx(0.001)
        assert resp.x_gateway_latency_ms == 250


class TestChatCompletionChunk:
    def test_id_starts_with_chatcmpl(self):
        chunk = ChatCompletionChunk(
            model="claude-haiku-4-5",
            choices=[
                StreamChoice(
                    index=0,
                    delta=ChoiceDelta(role="assistant", content="Hi"),
                )
            ],
        )
        assert chunk.id.startswith("chatcmpl-")

    def test_object_is_chunk(self):
        chunk = ChatCompletionChunk(
            model="gpt-4o",
            choices=[StreamChoice(index=0, delta=ChoiceDelta(content=" world"))],
        )
        assert chunk.object == "chat.completion.chunk"

    def test_usage_defaults_to_none(self):
        chunk = ChatCompletionChunk(
            model="gpt-4o",
            choices=[StreamChoice(index=0, delta=ChoiceDelta(content="Hi"))],
        )
        assert chunk.usage is None


class TestModelObject:
    def test_owned_by_default(self):
        m = ModelObject(id="claude-haiku-4-5")
        assert m.owned_by == "llm-gateway"
        assert m.object == "model"

    def test_custom_owned_by(self):
        m = ModelObject(id="gpt-4o", owned_by="openai")
        assert m.owned_by == "openai"


class TestModelsListResponse:
    def test_object_is_list(self):
        r = ModelsListResponse(data=[ModelObject(id="gpt-4o")])
        assert r.object == "list"

    def test_data_preserved(self):
        models = [ModelObject(id="a"), ModelObject(id="b")]
        r = ModelsListResponse(data=models)
        assert len(r.data) == 2


class TestProviderName:
    def test_all_providers(self):
        providers = {p.value for p in ProviderName}
        assert providers == {"anthropic", "openai", "ollama"}


class TestRoutingDecision:
    def test_valid(self):
        rd = RoutingDecision(
            provider=ProviderName.anthropic,
            model="claude-haiku-4-5",
            reason="cost",
            original_model="claude-haiku-4-5",
        )
        assert rd.provider == ProviderName.anthropic
        assert rd.fallback_chain == []


class TestCacheResult:
    def test_cache_miss_defaults(self):
        cr = CacheResult(hit=False)
        assert cr.hit is False
        assert cr.response is None
        assert cr.similarity == 0.0
        assert cr.cache_key is None

    def test_cache_hit(self):
        response = ChatCompletionResponse(
            model="gpt-4o",
            choices=[
                Choice(
                    index=0,
                    message=ChoiceMessage(role="assistant", content="cached"),
                    finish_reason="stop",
                )
            ],
        )
        cr = CacheResult(hit=True, response=response, similarity=0.95, cache_key="abc123")
        assert cr.hit is True
        assert cr.similarity == pytest.approx(0.95)
        assert cr.cache_key == "abc123"


class TestCostRecord:
    def test_valid(self):
        cr = CostRecord(
            request_id="req-1",
            api_key_hash="hash123",
            provider="anthropic",
            model="claude-haiku-4-5",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cost_usd=0.001,
            latency_ms=300,
            cache_hit=False,
        )
        assert cr.request_id == "req-1"
        assert cr.cache_hit is False
        assert cr.timestamp > 0


class TestRateLimitInfo:
    def test_allowed(self):
        rl = RateLimitInfo(
            allowed=True,
            remaining_requests=99,
            remaining_tokens=9000,
            reset_at=time.time() + 60,
        )
        assert rl.allowed is True
        assert rl.retry_after is None

    def test_denied_with_retry_after(self):
        rl = RateLimitInfo(
            allowed=False,
            remaining_requests=0,
            remaining_tokens=0,
            reset_at=time.time() + 30,
            retry_after=30.0,
        )
        assert rl.allowed is False
        assert rl.retry_after == pytest.approx(30.0)


class TestHealthStatus:
    def test_healthy(self):
        hs = HealthStatus(provider="anthropic", healthy=True, latency_ms=120.5)
        assert hs.healthy is True
        assert hs.latency_ms == pytest.approx(120.5)
        assert hs.error is None

    def test_unhealthy_with_error(self):
        hs = HealthStatus(provider="openai", healthy=False, error="Connection refused")
        assert hs.healthy is False
        assert hs.error == "Connection refused"


class TestGatewayStats:
    def test_valid(self):
        gs = GatewayStats(
            total_requests=1000,
            cache_hit_rate=0.35,
            total_cost_usd=1.5,
            requests_by_provider={"anthropic": 600, "openai": 400},
            avg_latency_ms=280.0,
            error_rate=0.01,
            uptime_seconds=86400.0,
        )
        assert gs.total_requests == 1000
        assert gs.cache_hit_rate == pytest.approx(0.35)


class TestErrorResponse:
    def test_create_factory(self):
        err = ErrorResponse.create("Something went wrong")
        assert err.error["message"] == "Something went wrong"
        assert err.error["type"] == "invalid_request_error"
        assert err.error["code"] is None
        assert err.error["param"] is None

    def test_create_with_custom_type_and_code(self):
        err = ErrorResponse.create("Auth failed", type_="authentication_error", code="401")
        assert err.error["type"] == "authentication_error"
        assert err.error["code"] == "401"

    def test_error_is_dict(self):
        err = ErrorResponse.create("Bad request")
        assert isinstance(err.error, dict)


class TestResponseFormat:
    def test_default_type_is_text(self):
        rf = ResponseFormat()
        assert rf.type == "text"

    def test_json_object_type(self):
        rf = ResponseFormat(type="json_object")
        assert rf.type == "json_object"


class TestStreamOptions:
    def test_include_usage_defaults_false(self):
        so = StreamOptions()
        assert so.include_usage is False

    def test_include_usage_true(self):
        so = StreamOptions(include_usage=True)
        assert so.include_usage is True
