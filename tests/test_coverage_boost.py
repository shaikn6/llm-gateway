"""Targeted tests to push coverage from ~80% to 95%+.

Covers uncovered branches in:
- src/providers/anthropic.py  (stream, _convert_messages tool/tool_calls, APIError, _build_kwargs full branches)
- src/providers/openai.py     (stream, APIError path)
- src/providers/ollama.py     (stream, empty content, raise_for_status branch)
- src/providers/openai_provider.py (sync complete)
- src/providers/base.py       (is_healthy cache-hit path, health_check_interval)
- src/gateway/router.py       (get_router singleton, openai provider import branch)
- src/api/main.py             (get_router singleton, health endpoint extras)
- src/models/schemas.py       (ErrorResponse.create, effective_max_tokens, all validators)
- src/middleware/usage_tracker.py (claude-sonnet-4-6 cost calc, multi-model by_model)
- src/middleware/rate_limiter.py  (pipeline key check edge cases)
- src/cache/semantic_cache.py     (multi-message lists, empty list)
- src/config.py               (Settings custom overrides, multiple env vars)
"""

from __future__ import annotations

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(**kwargs):
    from src.models.schemas import ChatCompletionRequest, ChatMessage, MessageRole

    defaults = dict(
        model="claude-3-haiku-20240307",
        messages=[ChatMessage(role=MessageRole.user, content="Hello")],
    )
    defaults.update(kwargs)
    return ChatCompletionRequest(**defaults)


def _make_openai_request(**kwargs):
    from src.models.schemas import ChatCompletionRequest

    defaults = dict(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello"}],
    )
    defaults.update(kwargs)
    return ChatCompletionRequest(**defaults)


# ---------------------------------------------------------------------------
# src/models/schemas.py — uncovered paths
# ---------------------------------------------------------------------------


class TestChatCompletionRequestValidation:
    def test_empty_messages_raises_validation_error(self):
        from pydantic import ValidationError
        from src.models.schemas import ChatCompletionRequest

        with pytest.raises(ValidationError, match="messages must not be empty"):
            ChatCompletionRequest(model="gpt-4o", messages=[])

    def test_effective_max_tokens_prefers_max_completion_tokens(self):
        from src.models.schemas import ChatCompletionRequest

        req = ChatCompletionRequest(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=512,
            max_completion_tokens=1024,
        )
        assert req.effective_max_tokens() == 1024

    def test_effective_max_tokens_falls_back_to_max_tokens(self):
        from src.models.schemas import ChatCompletionRequest

        req = ChatCompletionRequest(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=256,
        )
        assert req.effective_max_tokens() == 256

    def test_effective_max_tokens_returns_none_when_both_unset(self):
        from src.models.schemas import ChatCompletionRequest

        req = ChatCompletionRequest(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert req.effective_max_tokens() is None

    def test_temperature_out_of_range_raises(self):
        from pydantic import ValidationError
        from src.models.schemas import ChatCompletionRequest

        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="gpt-4o",
                messages=[{"role": "user", "content": "hi"}],
                temperature=3.0,
            )

    def test_top_p_out_of_range_raises(self):
        from pydantic import ValidationError
        from src.models.schemas import ChatCompletionRequest

        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="gpt-4o",
                messages=[{"role": "user", "content": "hi"}],
                top_p=1.5,
            )

    def test_n_out_of_range_raises(self):
        from pydantic import ValidationError
        from src.models.schemas import ChatCompletionRequest

        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="gpt-4o",
                messages=[{"role": "user", "content": "hi"}],
                n=10,
            )

    def test_x_provider_alias_roundtrip(self):
        from src.models.schemas import ChatCompletionRequest

        req = ChatCompletionRequest(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            **{"x-provider": "anthropic"},
        )
        assert req.x_provider == "anthropic"

    def test_stream_options_include_usage(self):
        from src.models.schemas import StreamOptions

        so = StreamOptions(include_usage=True)
        assert so.include_usage is True

    def test_response_format_json_object(self):
        from src.models.schemas import ResponseFormat

        rf = ResponseFormat(type="json_object")
        assert rf.type == "json_object"


class TestErrorResponseCreate:
    def test_create_produces_error_key(self):
        from src.models.schemas import ErrorResponse

        resp = ErrorResponse.create("Something went wrong")
        assert "error" in resp.model_dump()

    def test_create_default_type(self):
        from src.models.schemas import ErrorResponse

        resp = ErrorResponse.create("bad input")
        assert resp.error["type"] == "invalid_request_error"

    def test_create_custom_type(self):
        from src.models.schemas import ErrorResponse

        resp = ErrorResponse.create("auth failure", type_="authentication_error")
        assert resp.error["type"] == "authentication_error"

    def test_create_with_code(self):
        from src.models.schemas import ErrorResponse

        resp = ErrorResponse.create("Not found", code="model_not_found")
        assert resp.error["code"] == "model_not_found"

    def test_create_message_stored(self):
        from src.models.schemas import ErrorResponse

        resp = ErrorResponse.create("Access denied")
        assert resp.error["message"] == "Access denied"

    def test_create_param_is_none(self):
        from src.models.schemas import ErrorResponse

        resp = ErrorResponse.create("test")
        assert resp.error["param"] is None


class TestSchemaModels:
    def test_models_list_response(self):
        from src.models.schemas import ModelsListResponse, ModelObject

        obj = ModelObject(id="gpt-4o", owned_by="openai")
        resp = ModelsListResponse(data=[obj])
        assert resp.object == "list"
        assert len(resp.data) == 1

    def test_routing_decision_model(self):
        from src.models.schemas import RoutingDecision, ProviderName

        rd = RoutingDecision(
            provider=ProviderName.anthropic,
            model="claude-3-haiku-20240307",
            reason="model prefix match",
            original_model="claude-3-haiku-20240307",
        )
        assert rd.provider == ProviderName.anthropic
        assert rd.fallback_chain == []

    def test_cache_result_miss(self):
        from src.models.schemas import CacheResult

        cr = CacheResult(hit=False)
        assert cr.hit is False
        assert cr.response is None
        assert cr.similarity == 0.0

    def test_rate_limit_info(self):
        from src.models.schemas import RateLimitInfo

        rli = RateLimitInfo(
            allowed=True,
            remaining_requests=50,
            remaining_tokens=10000,
            reset_at=time.time() + 60,
        )
        assert rli.allowed is True
        assert rli.retry_after is None

    def test_gateway_stats_model(self):
        from src.models.schemas import GatewayStats

        gs = GatewayStats(
            total_requests=100,
            cache_hit_rate=0.35,
            total_cost_usd=1.23,
            requests_by_provider={"anthropic": 80, "openai": 20},
            avg_latency_ms=250.0,
            error_rate=0.01,
            uptime_seconds=3600.0,
        )
        assert gs.total_requests == 100
        assert gs.cache_hit_rate == 0.35

    def test_cost_record(self):
        from src.models.schemas import CostRecord

        cr = CostRecord(
            request_id="req-1",
            api_key_hash="abc123",
            provider="anthropic",
            model="claude-3-haiku-20240307",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cost_usd=0.0005,
            latency_ms=300,
            cache_hit=False,
        )
        assert cr.provider == "anthropic"
        assert cr.cache_hit is False

    def test_chat_completion_response_defaults(self):
        from src.models.schemas import ChatCompletionResponse, Choice, ChoiceMessage

        resp = ChatCompletionResponse(
            model="gpt-4o",
            choices=[
                Choice(
                    index=0,
                    message=ChoiceMessage(role="assistant", content="Hi"),
                )
            ],
        )
        assert resp.object == "chat.completion"
        assert resp.id.startswith("chatcmpl-")

    def test_chat_completion_chunk_defaults(self):
        from src.models.schemas import ChatCompletionChunk, StreamChoice, ChoiceDelta

        chunk = ChatCompletionChunk(
            model="gpt-4o",
            choices=[
                StreamChoice(
                    index=0,
                    delta=ChoiceDelta(role="assistant", content="hi"),
                )
            ],
        )
        assert chunk.object == "chat.completion.chunk"

    def test_usage_info_total_tokens(self):
        from src.models.schemas import UsageInfo

        u = UsageInfo(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        assert u.total_tokens == 30

    def test_health_status_with_error(self):
        from src.models.schemas import HealthStatus

        hs = HealthStatus(provider="openai", healthy=False, error="timeout")
        assert hs.healthy is False
        assert hs.error == "timeout"

    def test_provider_name_enum_values(self):
        from src.models.schemas import ProviderName

        assert ProviderName.anthropic == "anthropic"
        assert ProviderName.openai == "openai"
        assert ProviderName.ollama == "ollama"

    def test_message_role_enum_values(self):
        from src.models.schemas import MessageRole

        assert MessageRole.system == "system"
        assert MessageRole.tool == "tool"
        assert MessageRole.function == "function"


# ---------------------------------------------------------------------------
# src/providers/base.py — is_healthy cache-hit path
# ---------------------------------------------------------------------------


class TestLLMProviderIsHealthyCachePath:
    @pytest.mark.asyncio
    async def test_is_healthy_uses_cached_value_when_fresh(self):
        """When _last_health_check is recent, health_check is NOT called again."""
        from src.models.schemas import HealthStatus
        from src.providers.base import LLMProvider

        class ConcreteProvider(LLMProvider):
            name = "test"
            health_check_calls = 0

            async def complete(self, req): ...
            async def stream(self, req): ...
            async def list_models(self): ...

            async def health_check(self):
                self.health_check_calls += 1
                return HealthStatus(provider="test", healthy=True)

        p = ConcreteProvider(api_key="key")
        # Simulate a very recent health check
        p._last_health_check = time.monotonic()  # just now
        p._healthy = True

        result = await p.is_healthy()
        assert result is True
        assert p.health_check_calls == 0  # should NOT re-probe

    @pytest.mark.asyncio
    async def test_is_healthy_updates_cached_value_when_stale(self):
        from src.models.schemas import HealthStatus
        from src.providers.base import LLMProvider

        class ConcreteProvider(LLMProvider):
            name = "test"
            async def complete(self, req): ...
            async def stream(self, req): ...
            async def list_models(self): ...
            async def health_check(self):
                return HealthStatus(provider="test", healthy=False)

        p = ConcreteProvider(api_key="key")
        p._last_health_check = 0  # force stale
        p._healthy = True  # old cached value

        result = await p.is_healthy()
        assert result is False  # updated from health_check()

    def test_health_check_interval_default(self):
        from src.providers.base import LLMProvider

        class ConcreteProvider(LLMProvider):
            name = "test"
            async def complete(self, req): ...
            async def stream(self, req): ...
            async def list_models(self): ...
            async def health_check(self): ...

        p = ConcreteProvider(api_key="key")
        assert p._health_check_interval == 30.0


# ---------------------------------------------------------------------------
# src/providers/anthropic.py — stream, tool conversions, APIError
# ---------------------------------------------------------------------------


class TestAnthropicProviderStream:
    @pytest.mark.asyncio
    async def test_stream_yields_content_block_delta_chunks(self):
        import anthropic
        from src.providers.anthropic import AnthropicProvider
        from src.models.schemas import ChatCompletionChunk

        event1 = MagicMock()
        event1.type = "content_block_delta"
        event1.delta = MagicMock(text="Hello ")

        event2 = MagicMock()
        event2.type = "content_block_delta"
        event2.delta = MagicMock(text="world")

        event3 = MagicMock()
        event3.type = "message_stop"

        async def _fake_aiter():
            for e in [event1, event2, event3]:
                yield e

        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream.__aexit__ = AsyncMock(return_value=False)
        mock_stream.__aiter__ = lambda self: _fake_aiter()

        mock_client = AsyncMock()
        mock_client.messages.stream = MagicMock(return_value=mock_stream)

        with patch("src.providers.anthropic.anthropic.AsyncAnthropic", return_value=mock_client):
            provider = AnthropicProvider(api_key="sk-ant-test")
        provider._client = mock_client

        req = _make_request()
        chunks = []
        async for chunk in provider.stream(req):
            chunks.append(chunk)

        assert len(chunks) == 3
        assert isinstance(chunks[0], ChatCompletionChunk)
        assert chunks[0].choices[0].delta.content == "Hello "
        assert chunks[1].choices[0].delta.content == "world"
        assert chunks[2].choices[0].finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_stream_raises_auth_error(self):
        import anthropic
        from src.providers.anthropic import AnthropicProvider
        from src.providers.base import ProviderAuthError

        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(
            side_effect=anthropic.AuthenticationError(
                message="bad key",
                response=MagicMock(status_code=401),
                body={},
            )
        )
        mock_stream.__aexit__ = AsyncMock(return_value=False)

        mock_client = AsyncMock()
        mock_client.messages.stream = MagicMock(return_value=mock_stream)

        with patch("src.providers.anthropic.anthropic.AsyncAnthropic", return_value=mock_client):
            provider = AnthropicProvider(api_key="sk-ant-test")
        provider._client = mock_client

        with pytest.raises(ProviderAuthError):
            async for _ in provider.stream(_make_request()):
                pass

    @pytest.mark.asyncio
    async def test_stream_raises_rate_limit_error(self):
        import anthropic
        from src.providers.anthropic import AnthropicProvider
        from src.providers.base import ProviderRateLimitError

        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(
            side_effect=anthropic.RateLimitError(
                message="rate limited",
                response=MagicMock(status_code=429),
                body={},
            )
        )
        mock_stream.__aexit__ = AsyncMock(return_value=False)

        mock_client = AsyncMock()
        mock_client.messages.stream = MagicMock(return_value=mock_stream)

        with patch("src.providers.anthropic.anthropic.AsyncAnthropic", return_value=mock_client):
            provider = AnthropicProvider(api_key="sk-ant-test")
        provider._client = mock_client

        with pytest.raises(ProviderRateLimitError):
            async for _ in provider.stream(_make_request()):
                pass

    @pytest.mark.asyncio
    async def test_stream_raises_timeout_error(self):
        import anthropic
        from src.providers.anthropic import AnthropicProvider
        from src.providers.base import ProviderTimeoutError

        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(
            side_effect=anthropic.APITimeoutError(request=MagicMock())
        )
        mock_stream.__aexit__ = AsyncMock(return_value=False)

        mock_client = AsyncMock()
        mock_client.messages.stream = MagicMock(return_value=mock_stream)

        with patch("src.providers.anthropic.anthropic.AsyncAnthropic", return_value=mock_client):
            provider = AnthropicProvider(api_key="sk-ant-test")
        provider._client = mock_client

        with pytest.raises(ProviderTimeoutError):
            async for _ in provider.stream(_make_request()):
                pass

    @pytest.mark.asyncio
    async def test_stream_raises_api_error(self):
        import anthropic
        from src.providers.anthropic import AnthropicProvider
        from src.providers.base import ProviderError

        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(
            side_effect=anthropic.APIError(
                message="internal",
                request=MagicMock(),
                body={},
            )
        )
        mock_stream.__aexit__ = AsyncMock(return_value=False)

        mock_client = AsyncMock()
        mock_client.messages.stream = MagicMock(return_value=mock_stream)

        with patch("src.providers.anthropic.anthropic.AsyncAnthropic", return_value=mock_client):
            provider = AnthropicProvider(api_key="sk-ant-test")
        provider._client = mock_client

        with pytest.raises(ProviderError):
            async for _ in provider.stream(_make_request()):
                pass


class TestConvertMessagesExtended:
    """Additional _convert_messages branches not covered by existing tests."""

    def test_tool_message_converted_to_tool_result(self):
        from src.providers.anthropic import _convert_messages
        from src.models.schemas import ChatCompletionRequest, ChatMessage, MessageRole

        req = ChatCompletionRequest(
            model="claude-3-haiku-20240307",
            messages=[
                ChatMessage(role=MessageRole.user, content="Use the tool"),
                ChatMessage(
                    role=MessageRole.tool,
                    content="42",
                    tool_call_id="call_abc",
                ),
            ],
        )
        _, messages = _convert_messages(req)
        tool_result_msg = messages[1]
        assert tool_result_msg["role"] == "user"
        assert tool_result_msg["content"][0]["type"] == "tool_result"
        assert tool_result_msg["content"][0]["tool_use_id"] == "call_abc"

    def test_assistant_with_tool_calls_produces_tool_use_blocks(self):
        from src.providers.anthropic import _convert_messages
        from src.models.schemas import ChatCompletionRequest, ChatMessage, MessageRole

        req = ChatCompletionRequest(
            model="claude-3-haiku-20240307",
            messages=[
                ChatMessage(role=MessageRole.user, content="Call the tool"),
                ChatMessage(
                    role=MessageRole.assistant,
                    content=None,
                    tool_calls=[
                        {
                            "id": "call_123",
                            "function": {
                                "name": "get_weather",
                                "arguments": {"location": "NYC"},
                            },
                        }
                    ],
                ),
            ],
        )
        _, messages = _convert_messages(req)
        asst_msg = messages[1]
        assert asst_msg["role"] == "assistant"
        assert isinstance(asst_msg["content"], list)
        tool_use = asst_msg["content"][0]
        assert tool_use["type"] == "tool_use"
        assert tool_use["name"] == "get_weather"

    def test_list_content_passed_through(self):
        """When content is a list (multimodal), it should be passed as-is."""
        from src.providers.anthropic import _convert_messages
        from src.models.schemas import ChatCompletionRequest, ChatMessage, MessageRole

        content_list = [
            {"type": "text", "text": "Hello"},
            {"type": "image_url", "image_url": {"url": "http://ex.com/img.png"}},
        ]
        req = ChatCompletionRequest(
            model="claude-3-haiku-20240307",
            messages=[ChatMessage(role=MessageRole.user, content=content_list)],
        )
        _, messages = _convert_messages(req)
        assert messages[0]["content"] == content_list

    def test_function_role_treated_as_assistant(self):
        """The function role should map to assistant (not system/tool)."""
        from src.providers.anthropic import _convert_messages
        from src.models.schemas import ChatCompletionRequest, ChatMessage, MessageRole

        req = ChatCompletionRequest(
            model="claude-3-haiku-20240307",
            messages=[
                ChatMessage(role=MessageRole.user, content="hi"),
                ChatMessage(role=MessageRole.function, content="result"),
            ],
        )
        _, messages = _convert_messages(req)
        assert messages[1]["role"] == "assistant"


class TestAnthropicBuildKwargs:
    """Cover _build_kwargs branches for top_p, temperature, stop list."""

    @pytest.mark.asyncio
    async def test_top_p_is_passed_when_set(self):
        from src.providers.anthropic import AnthropicProvider

        mock_resp = MagicMock()
        mock_resp.id = "msg_1"
        mock_resp.model = "claude-3-haiku-20240307"
        mock_resp.stop_reason = "end_turn"
        block = MagicMock()
        block.text = "ok"
        mock_resp.content = [block]
        mock_resp.usage = MagicMock(input_tokens=5, output_tokens=3)

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_resp)

        with patch("src.providers.anthropic.anthropic.AsyncAnthropic", return_value=mock_client):
            provider = AnthropicProvider(api_key="sk-ant-test")
        provider._client = mock_client

        req = _make_request(top_p=0.9)
        await provider.complete(req)
        kwargs = mock_client.messages.create.call_args[1]
        assert kwargs["top_p"] == 0.9

    @pytest.mark.asyncio
    async def test_temperature_is_passed_when_set(self):
        from src.providers.anthropic import AnthropicProvider

        mock_resp = MagicMock()
        mock_resp.id = "msg_2"
        mock_resp.model = "claude-3-haiku-20240307"
        mock_resp.stop_reason = "end_turn"
        block = MagicMock()
        block.text = "ok"
        mock_resp.content = [block]
        mock_resp.usage = MagicMock(input_tokens=5, output_tokens=3)

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_resp)

        with patch("src.providers.anthropic.anthropic.AsyncAnthropic", return_value=mock_client):
            provider = AnthropicProvider(api_key="sk-ant-test")
        provider._client = mock_client

        req = _make_request(temperature=0.3)
        await provider.complete(req)
        kwargs = mock_client.messages.create.call_args[1]
        assert kwargs["temperature"] == 0.3

    @pytest.mark.asyncio
    async def test_complete_raises_api_error(self):
        import anthropic
        from src.providers.anthropic import AnthropicProvider
        from src.providers.base import ProviderError

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(
            side_effect=anthropic.APIError(
                message="server error",
                request=MagicMock(),
                body={},
            )
        )

        with patch("src.providers.anthropic.anthropic.AsyncAnthropic", return_value=mock_client):
            provider = AnthropicProvider(api_key="sk-ant-test")
        provider._client = mock_client

        with pytest.raises(ProviderError):
            await provider.complete(_make_request())

    @pytest.mark.asyncio
    async def test_complete_multiple_content_blocks_concatenated(self):
        """When response has multiple content blocks, their text is concatenated."""
        from src.providers.anthropic import AnthropicProvider

        block1 = MagicMock()
        block1.text = "Hello "
        block2 = MagicMock()
        block2.text = "world"
        block_no_text = MagicMock(spec=[])  # no .text attribute

        mock_resp = MagicMock()
        mock_resp.id = "msg_multi"
        mock_resp.model = "claude-3-haiku-20240307"
        mock_resp.stop_reason = "end_turn"
        mock_resp.content = [block1, block2, block_no_text]
        mock_resp.usage = MagicMock(input_tokens=5, output_tokens=10)

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_resp)

        with patch("src.providers.anthropic.anthropic.AsyncAnthropic", return_value=mock_client):
            provider = AnthropicProvider(api_key="sk-ant-test")
        provider._client = mock_client

        resp = await provider.complete(_make_request())
        assert resp.choices[0].message.content == "Hello world"


# ---------------------------------------------------------------------------
# src/providers/openai.py — stream path and APIError branch
# ---------------------------------------------------------------------------


class TestOpenAIAsyncProviderStream:
    @pytest.mark.asyncio
    async def test_stream_yields_chunks(self):
        from src.providers.openai import OpenAIAsyncProvider
        from src.models.schemas import ChatCompletionChunk

        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta.content = "Hello"
        chunk1.choices[0].finish_reason = None

        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta.content = None
        chunk2.choices[0].finish_reason = "stop"

        async def _fake_aiter():
            for c in [chunk1, chunk2]:
                yield c

        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream.__aexit__ = AsyncMock(return_value=False)
        mock_stream.__aiter__ = lambda self: _fake_aiter()

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)

        with patch("src.providers.openai.openai_lib.AsyncOpenAI", return_value=mock_client):
            provider = OpenAIAsyncProvider(api_key="sk-test")
        provider._client = mock_client

        req = _make_openai_request()
        chunks = []
        async for chunk in provider.stream(req):
            chunks.append(chunk)

        assert len(chunks) == 2
        assert isinstance(chunks[0], ChatCompletionChunk)
        assert chunks[0].choices[0].delta.content == "Hello"
        assert chunks[1].choices[0].finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_stream_raises_provider_error_on_api_error(self):
        import openai
        from src.providers.openai import OpenAIAsyncProvider
        from src.providers.base import ProviderError

        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(
            side_effect=openai.APIError(
                message="stream error",
                request=MagicMock(),
                body={},
            )
        )
        mock_stream.__aexit__ = AsyncMock(return_value=False)

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)

        with patch("src.providers.openai.openai_lib.AsyncOpenAI", return_value=mock_client):
            provider = OpenAIAsyncProvider(api_key="sk-test")
        provider._client = mock_client

        with pytest.raises(ProviderError):
            async for _ in provider.stream(_make_openai_request()):
                pass

    @pytest.mark.asyncio
    async def test_stream_empty_choices_handled(self):
        """When chunk.choices is empty, delta and finish_reason should be None."""
        from src.providers.openai import OpenAIAsyncProvider

        chunk_empty = MagicMock()
        chunk_empty.choices = []

        async def _fake_aiter():
            yield chunk_empty

        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream.__aexit__ = AsyncMock(return_value=False)
        mock_stream.__aiter__ = lambda self: _fake_aiter()

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)

        with patch("src.providers.openai.openai_lib.AsyncOpenAI", return_value=mock_client):
            provider = OpenAIAsyncProvider(api_key="sk-test")
        provider._client = mock_client

        chunks = []
        async for chunk in provider.stream(_make_openai_request()):
            chunks.append(chunk)
        assert len(chunks) == 1
        assert chunks[0].choices[0].delta.content is None
        assert chunks[0].choices[0].finish_reason is None


# ---------------------------------------------------------------------------
# src/providers/ollama.py — stream path
# ---------------------------------------------------------------------------


class TestOllamaProviderStream:
    @pytest.mark.asyncio
    async def test_stream_yields_chunks_until_done(self):
        from src.providers.ollama import OllamaProvider
        from src.models.schemas import ChatCompletionChunk

        lines = [
            json.dumps({"message": {"content": "Hello"}, "done": False}),
            json.dumps({"message": {"content": " world"}, "done": True}),
        ]

        async def _fake_aiter_lines():
            for line in lines:
                yield line

        mock_resp = AsyncMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.aiter_lines = _fake_aiter_lines

        mock_stream_ctx = AsyncMock()
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=mock_stream_ctx)

        with patch("src.providers.ollama.httpx.AsyncClient", return_value=mock_client):
            provider = OllamaProvider()
        provider._client = mock_client

        req = _make_request(model="llama3")
        chunks = []
        async for chunk in provider.stream(req):
            chunks.append(chunk)

        assert len(chunks) == 2
        assert isinstance(chunks[0], ChatCompletionChunk)
        assert chunks[0].choices[0].delta.content == "Hello"
        assert chunks[0].choices[0].finish_reason is None
        assert chunks[1].choices[0].finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_stream_skips_empty_lines(self):
        from src.providers.ollama import OllamaProvider

        lines = [
            "",  # empty — should be skipped
            json.dumps({"message": {"content": "text"}, "done": True}),
        ]

        async def _fake_aiter_lines():
            for line in lines:
                yield line

        mock_resp = AsyncMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.aiter_lines = _fake_aiter_lines

        mock_stream_ctx = AsyncMock()
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=mock_stream_ctx)

        with patch("src.providers.ollama.httpx.AsyncClient", return_value=mock_client):
            provider = OllamaProvider()
        provider._client = mock_client

        chunks = []
        async for chunk in provider.stream(_make_request(model="llama3")):
            chunks.append(chunk)
        assert len(chunks) == 1  # empty line skipped

    @pytest.mark.asyncio
    async def test_stream_raises_provider_error_on_http_error(self):
        import httpx
        from src.providers.ollama import OllamaProvider
        from src.providers.base import ProviderError

        mock_stream_ctx = AsyncMock()
        mock_stream_ctx.__aenter__ = AsyncMock(
            side_effect=httpx.HTTPError("connection refused")
        )
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=mock_stream_ctx)

        with patch("src.providers.ollama.httpx.AsyncClient", return_value=mock_client):
            provider = OllamaProvider()
        provider._client = mock_client

        with pytest.raises(ProviderError):
            async for _ in provider.stream(_make_request(model="llama3")):
                pass


# ---------------------------------------------------------------------------
# src/providers/openai_provider.py — sync OpenAIProvider.complete
# ---------------------------------------------------------------------------


class TestOpenAIProviderSync:
    def test_complete_returns_dict_with_expected_keys(self):
        """The lightweight sync OpenAIProvider used by GatewayRouter."""
        mock_resp = MagicMock()
        mock_resp.id = "chatcmpl-sync-1"
        mock_resp.model = "gpt-4o-mini"
        choice = MagicMock()
        choice.message.content = "Sync response"
        mock_resp.choices = [choice]
        mock_resp.usage = MagicMock(prompt_tokens=5, completion_tokens=3)

        mock_openai_client = MagicMock()
        mock_openai_client.chat.completions.create.return_value = mock_resp

        with patch("src.providers.openai_provider.openai.OpenAI", return_value=mock_openai_client):
            from src.providers.openai_provider import OpenAIProvider
            provider = OpenAIProvider(api_key="sk-test")

        result = provider.complete(
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-4o-mini",
        )
        assert result["id"] == "chatcmpl-sync-1"
        assert result["model"] == "gpt-4o-mini"
        assert result["choices"][0]["message"]["content"] == "Sync response"
        assert "prompt_tokens" in result["usage"]

    def test_complete_passes_kwargs_to_create(self):
        mock_resp = MagicMock()
        mock_resp.id = "id"
        mock_resp.model = "gpt-4o-mini"
        choice = MagicMock()
        choice.message.content = "ok"
        mock_resp.choices = [choice]
        mock_resp.usage = MagicMock(prompt_tokens=1, completion_tokens=1)

        mock_openai_client = MagicMock()
        mock_openai_client.chat.completions.create.return_value = mock_resp

        with patch("src.providers.openai_provider.openai.OpenAI", return_value=mock_openai_client):
            from src.providers.openai_provider import OpenAIProvider
            provider = OpenAIProvider(api_key="sk-test")

        provider.complete(
            messages=[{"role": "user", "content": "test"}],
            model="gpt-4o-mini",
            max_tokens=100,
        )
        call_kwargs = mock_openai_client.chat.completions.create.call_args[1]
        assert call_kwargs["max_tokens"] == 100


# ---------------------------------------------------------------------------
# src/gateway/router.py — GatewayRouter and get_router
# ---------------------------------------------------------------------------


class TestGetRouterSingleton:
    def test_get_router_returns_a_router_instance(self):
        """get_router() should return a GatewayRouter (creating one if needed)."""
        import src.api.main as main_module
        from src.gateway.router import GatewayRouter

        original = main_module._gateway
        try:
            main_module._gateway = None
            with patch.object(main_module, "settings") as mock_settings:
                mock_settings.anthropic_api_key = "sk-ant-test"
                mock_settings.openai_api_key = "sk-openai-test"
                result = main_module.get_router()
            assert isinstance(result, GatewayRouter)
        finally:
            main_module._gateway = original

    def test_get_router_returns_same_instance_on_second_call(self):
        """Singleton: second call returns same object."""
        import src.api.main as main_module

        original = main_module._gateway
        try:
            main_module._gateway = None
            with patch.object(main_module, "settings") as mock_settings:
                mock_settings.anthropic_api_key = "sk-ant-test"
                mock_settings.openai_api_key = "sk-openai-test"
                r1 = main_module.get_router()
                r2 = main_module.get_router()
            assert r1 is r2
        finally:
            main_module._gateway = original

    def test_health_endpoint_returns_version(self):
        from fastapi.testclient import TestClient
        from src.api.main import app

        client = TestClient(app)
        resp = client.get("/health")
        assert resp.json()["version"] == "0.1.0"


# ---------------------------------------------------------------------------
# src/config.py — extended Settings edge cases
# ---------------------------------------------------------------------------


class TestSettingsExtended:
    def test_custom_rate_limit_requests(self):
        from src.config import Settings

        with patch.dict("os.environ", {"RATE_LIMIT_REQUESTS": "50"}):
            s = Settings()
        assert s.rate_limit_requests == 50

    def test_custom_rate_limit_window(self):
        from src.config import Settings

        with patch.dict("os.environ", {"RATE_LIMIT_WINDOW_S": "120"}):
            s = Settings()
        assert s.rate_limit_window_s == 120

    def test_custom_cache_ttl(self):
        from src.config import Settings

        with patch.dict("os.environ", {"CACHE_TTL_S": "7200"}):
            s = Settings()
        assert s.cache_ttl_s == 7200

    def test_custom_redis_url(self):
        from src.config import Settings

        with patch.dict("os.environ", {"REDIS_URL": "redis://myhost:6380/1"}):
            s = Settings()
        assert s.redis_url == "redis://myhost:6380/1"

    def test_custom_api_keys(self):
        from src.config import Settings

        with patch.dict("os.environ", {"API_KEYS": "key-a,key-b,key-c"}):
            s = Settings()
        assert "key-a" in s.api_keys
        assert "key-c" in s.api_keys

    def test_log_level_can_be_changed(self):
        from src.config import Settings

        with patch.dict("os.environ", {"LOG_LEVEL": "DEBUG"}):
            s = Settings()
        assert s.log_level == "DEBUG"

    def test_openai_api_key_from_env(self):
        from src.config import Settings

        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-openai-xyz"}):
            s = Settings()
        assert s.openai_api_key == "sk-openai-xyz"


# ---------------------------------------------------------------------------
# src/middleware/usage_tracker.py — cost calculation branches
# ---------------------------------------------------------------------------


class TestUsageTrackerCostBranches:
    def _make_entry(self, model, prompt_tokens, completion_tokens):
        return json.dumps(
            {
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "ts": 1234567890.0,
            }
        )

    def _make_tracker(self):
        mock_redis = MagicMock()
        with patch("src.middleware.usage_tracker.redis.from_url", return_value=mock_redis):
            from src.middleware.usage_tracker import UsageTracker
            ut = UsageTracker()
        return ut, mock_redis

    def test_cost_calculation_for_claude_sonnet(self):
        ut, mock_redis = self._make_tracker()
        entry = self._make_entry("claude-sonnet-4-6", 1_000_000, 1_000_000)
        mock_redis.lrange.return_value = [entry]
        result = ut.get_usage("key")
        expected_cost = (1_000_000 * 3.0 + 1_000_000 * 15.0) / 1_000_000
        assert abs(result["cost_usd"] - expected_cost) < 0.0001

    def test_cost_zero_for_completely_unknown_model(self):
        ut, mock_redis = self._make_tracker()
        entry = self._make_entry("completely-unknown-model-xyz", 1000, 500)
        mock_redis.lrange.return_value = [entry]
        result = ut.get_usage("key")
        assert result["cost_usd"] == 0.0

    def test_gpt4o_mini_cost_calculation(self):
        ut, mock_redis = self._make_tracker()
        entry = self._make_entry("gpt-4o-mini", 1_000_000, 1_000_000)
        mock_redis.lrange.return_value = [entry]
        result = ut.get_usage("key")
        expected = (1_000_000 * 0.15 + 1_000_000 * 0.6) / 1_000_000
        assert abs(result["cost_usd"] - expected) < 0.0001

    def test_by_model_accumulates_costs_across_entries(self):
        ut, mock_redis = self._make_tracker()
        entries = [
            self._make_entry("gpt-4o", 100, 50),
            self._make_entry("gpt-4o", 200, 100),
        ]
        mock_redis.lrange.return_value = entries
        result = ut.get_usage("key")
        assert "gpt-4o" in result["by_model"]
        expected_total = (
            (100 * 5.0 + 50 * 15.0) / 1_000_000 +
            (200 * 5.0 + 100 * 15.0) / 1_000_000
        )
        assert abs(result["by_model"]["gpt-4o"] - expected_total) < 0.000001


# ---------------------------------------------------------------------------
# src/middleware/rate_limiter.py — additional edge-case branches
# ---------------------------------------------------------------------------


class TestRateLimiterEdgeCases:
    def _make_limiter(self, limit=100, window_s=60):
        mock_redis = MagicMock()
        with patch("src.middleware.rate_limiter.redis.from_url", return_value=mock_redis):
            from src.middleware.rate_limiter import RateLimiter
            rl = RateLimiter(limit=limit, window_s=window_s)
        return rl, mock_redis

    def test_remaining_is_max_0_when_over_limit(self):
        rl, mock_redis = self._make_limiter(limit=10)
        pipe = MagicMock()
        pipe.execute.return_value = [None, 20]  # way over limit
        mock_redis.pipeline.return_value = pipe

        allowed, remaining = rl.check("key")
        assert allowed is False
        assert remaining == 0  # max(0, 10 - 20 - 1) => 0

    def test_custom_window_used_in_pipeline(self):
        rl, mock_redis = self._make_limiter(window_s=120)
        pipe = MagicMock()
        pipe.execute.return_value = [None, 0]
        mock_redis.pipeline.return_value = pipe

        rl.check("key")
        args = pipe.zremrangebyscore.call_args[0]
        assert args[0] == "ratelimit:key"

    def test_record_expire_uses_double_window(self):
        rl, mock_redis = self._make_limiter(window_s=30)
        rl.record("mykey")
        mock_redis.expire.assert_called_once_with("ratelimit:mykey", 60)  # 30 * 2

    def test_check_pipeline_executes(self):
        rl, mock_redis = self._make_limiter()
        pipe = MagicMock()
        pipe.execute.return_value = [None, 5]
        mock_redis.pipeline.return_value = pipe

        rl.check("key")
        pipe.execute.assert_called_once()


# ---------------------------------------------------------------------------
# src/cache/semantic_cache.py — additional branches
# ---------------------------------------------------------------------------


class TestSemanticCacheAdditional:
    def _make_cache(self, ttl=3600):
        mock_redis = MagicMock()
        with patch("src.cache.semantic_cache.redis.from_url", return_value=mock_redis):
            from src.cache.semantic_cache import SemanticCache
            sc = SemanticCache(ttl_s=ttl)
        return sc, mock_redis

    def test_get_with_empty_messages_list(self):
        sc, mock_redis = self._make_cache()
        mock_redis.get.return_value = None
        result = sc.get([])
        assert result is None

    def test_set_with_complex_nested_response(self):
        sc, mock_redis = self._make_cache()
        messages = [{"role": "user", "content": "complex"}]
        response = {
            "id": "chatcmpl-xyz",
            "choices": [{"message": {"role": "assistant", "content": "yes"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3},
        }
        sc.set(messages, response)
        stored_json = mock_redis.setex.call_args[0][2]
        assert json.loads(stored_json) == response

    def test_key_consistency_across_multiple_calls(self):
        sc, _ = self._make_cache()
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        key1 = sc._key(messages)
        key2 = sc._key(messages)
        assert key1 == key2

    def test_get_returns_none_when_redis_returns_none(self):
        sc, mock_redis = self._make_cache()
        mock_redis.get.return_value = None
        result = sc.get([{"role": "user", "content": "test"}])
        assert result is None

    def test_set_serializes_response_as_json(self):
        sc, mock_redis = self._make_cache()
        response = {"key": "value", "nested": {"a": 1}}
        sc.set([{"role": "user", "content": "q"}], response)
        stored = mock_redis.setex.call_args[0][2]
        assert json.loads(stored) == response
