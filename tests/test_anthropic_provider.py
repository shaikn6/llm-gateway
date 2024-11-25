"""Tests for src/providers/anthropic.py — AnthropicProvider."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.providers.anthropic import AnthropicProvider, _convert_messages, ANTHROPIC_MODELS
from src.providers.base import (
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from src.models.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    MessageRole,
)


def _make_request(**kwargs):
    defaults = dict(
        model="claude-3-haiku-20240307",
        messages=[{"role": "user", "content": "Hello"}],
    )
    defaults.update(kwargs)
    return ChatCompletionRequest(**defaults)


def _make_mock_anthropic_response(text="Hello from Claude", input_tokens=10, output_tokens=20):
    resp = MagicMock()
    resp.id = "msg_test123"
    resp.model = "claude-3-haiku-20240307"
    resp.stop_reason = "end_turn"
    block = MagicMock()
    block.text = text
    resp.content = [block]
    resp.usage = MagicMock(input_tokens=input_tokens, output_tokens=output_tokens)
    return resp


class TestAnthropicProviderInit:
    def test_raises_auth_error_when_no_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ProviderAuthError):
                AnthropicProvider(api_key=None)

    def test_accepts_explicit_api_key(self):
        with patch("src.providers.anthropic.anthropic.AsyncAnthropic"):
            provider = AnthropicProvider(api_key="sk-ant-test")
        assert provider.api_key == "sk-ant-test"

    def test_accepts_key_from_env(self):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "env-key"}):
            with patch("src.providers.anthropic.anthropic.AsyncAnthropic"):
                provider = AnthropicProvider()
        assert provider.api_key == "env-key"

    def test_provider_name_is_anthropic(self):
        with patch("src.providers.anthropic.anthropic.AsyncAnthropic"):
            provider = AnthropicProvider(api_key="sk-ant-test")
        assert provider.name == "anthropic"


class TestConvertMessages:
    def test_system_message_extracted(self):
        req = _make_request(
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ]
        )
        system, messages = _convert_messages(req)
        assert system == "You are helpful."
        assert all(m["role"] != "system" for m in messages)

    def test_user_message_kept_as_user(self):
        req = _make_request(messages=[{"role": "user", "content": "Hi"}])
        system, messages = _convert_messages(req)
        assert system is None
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hi"

    def test_assistant_message_kept_as_assistant(self):
        req = _make_request(
            messages=[
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
            ]
        )
        _, messages = _convert_messages(req)
        assert messages[1]["role"] == "assistant"

    def test_tool_role_mapped_to_user_with_tool_result(self):
        req = _make_request(
            messages=[
                {"role": "user", "content": "Use tool"},
                {
                    "role": "tool",
                    "content": "tool result",
                    "tool_call_id": "tc1",
                },
            ]
        )
        _, messages = _convert_messages(req)
        # Tool message becomes user message with tool_result block
        tool_msgs = [m for m in messages if m["role"] == "user" and isinstance(m["content"], list)]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["content"][0]["type"] == "tool_result"

    def test_assistant_tool_calls_converted_to_tool_use(self):
        tool_calls = [
            {
                "id": "tc1",
                "function": {"name": "search", "arguments": '{"query": "test"}'},
            }
        ]
        req = _make_request(
            messages=[
                {"role": "user", "content": "Search"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tool_calls,
                },
            ]
        )
        _, messages = _convert_messages(req)
        assistant_msgs = [m for m in messages if m["role"] == "assistant"]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0]["content"][0]["type"] == "tool_use"
        assert assistant_msgs[0]["content"][0]["name"] == "search"

    def test_no_system_returns_none(self):
        req = _make_request(messages=[{"role": "user", "content": "Hello"}])
        system, _ = _convert_messages(req)
        assert system is None


class TestAnthropicProviderComplete:
    @pytest.mark.asyncio
    async def test_complete_returns_chat_completion_response(self):
        mock_resp = _make_mock_anthropic_response()
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_resp)

        with patch("src.providers.anthropic.anthropic.AsyncAnthropic", return_value=mock_client):
            provider = AnthropicProvider(api_key="sk-ant-test")
        provider._client = mock_client

        req = _make_request()
        resp = await provider.complete(req)

        assert isinstance(resp, ChatCompletionResponse)
        assert resp.choices[0].message.content == "Hello from Claude"

    @pytest.mark.asyncio
    async def test_complete_maps_usage_correctly(self):
        mock_resp = _make_mock_anthropic_response(input_tokens=100, output_tokens=200)
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_resp)

        with patch("src.providers.anthropic.anthropic.AsyncAnthropic", return_value=mock_client):
            provider = AnthropicProvider(api_key="sk-ant-test")
        provider._client = mock_client

        resp = await provider.complete(_make_request())
        assert resp.usage.prompt_tokens == 100
        assert resp.usage.completion_tokens == 200
        assert resp.usage.total_tokens == 300

    @pytest.mark.asyncio
    async def test_complete_raises_provider_auth_error_on_auth_failure(self):
        import anthropic
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(
            side_effect=anthropic.AuthenticationError(
                message="bad key",
                response=MagicMock(status_code=401),
                body={},
            )
        )

        with patch("src.providers.anthropic.anthropic.AsyncAnthropic", return_value=mock_client):
            provider = AnthropicProvider(api_key="sk-ant-bad")
        provider._client = mock_client

        with pytest.raises(ProviderAuthError):
            await provider.complete(_make_request())

    @pytest.mark.asyncio
    async def test_complete_raises_provider_rate_limit_error(self):
        import anthropic
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(
            side_effect=anthropic.RateLimitError(
                message="rate limited",
                response=MagicMock(status_code=429),
                body={},
            )
        )

        with patch("src.providers.anthropic.anthropic.AsyncAnthropic", return_value=mock_client):
            provider = AnthropicProvider(api_key="sk-ant-test")
        provider._client = mock_client

        with pytest.raises(ProviderRateLimitError):
            await provider.complete(_make_request())

    @pytest.mark.asyncio
    async def test_complete_raises_provider_timeout_error(self):
        import anthropic
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(
            side_effect=anthropic.APITimeoutError(request=MagicMock())
        )

        with patch("src.providers.anthropic.anthropic.AsyncAnthropic", return_value=mock_client):
            provider = AnthropicProvider(api_key="sk-ant-test")
        provider._client = mock_client

        with pytest.raises(ProviderTimeoutError):
            await provider.complete(_make_request())

    @pytest.mark.asyncio
    async def test_complete_builds_kwargs_with_system(self):
        mock_resp = _make_mock_anthropic_response()
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_resp)

        with patch("src.providers.anthropic.anthropic.AsyncAnthropic", return_value=mock_client):
            provider = AnthropicProvider(api_key="sk-ant-test")
        provider._client = mock_client

        req = _make_request(
            messages=[
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "Hi"},
            ]
        )
        await provider.complete(req)
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs.get("system") == "Be helpful."

    @pytest.mark.asyncio
    async def test_complete_stop_sequences_from_string(self):
        mock_resp = _make_mock_anthropic_response()
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_resp)

        with patch("src.providers.anthropic.anthropic.AsyncAnthropic", return_value=mock_client):
            provider = AnthropicProvider(api_key="sk-ant-test")
        provider._client = mock_client

        req = _make_request(stop="\n")
        await provider.complete(req)
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["stop_sequences"] == ["\n"]

    @pytest.mark.asyncio
    async def test_complete_stop_sequences_from_list(self):
        mock_resp = _make_mock_anthropic_response()
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_resp)

        with patch("src.providers.anthropic.anthropic.AsyncAnthropic", return_value=mock_client):
            provider = AnthropicProvider(api_key="sk-ant-test")
        provider._client = mock_client

        req = _make_request(stop=["\n", "END"])
        await provider.complete(req)
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["stop_sequences"] == ["\n", "END"]

    @pytest.mark.asyncio
    async def test_complete_id_has_chatcmpl_prefix(self):
        mock_resp = _make_mock_anthropic_response()
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_resp)

        with patch("src.providers.anthropic.anthropic.AsyncAnthropic", return_value=mock_client):
            provider = AnthropicProvider(api_key="sk-ant-test")
        provider._client = mock_client

        resp = await provider.complete(_make_request())
        assert resp.id.startswith("chatcmpl-")


class TestAnthropicProviderListModels:
    @pytest.mark.asyncio
    async def test_list_models_returns_all_supported_models(self):
        with patch("src.providers.anthropic.anthropic.AsyncAnthropic"):
            provider = AnthropicProvider(api_key="sk-ant-test")
        models = await provider.list_models()
        model_ids = {m.id for m in models}
        assert model_ids == set(ANTHROPIC_MODELS.keys())

    @pytest.mark.asyncio
    async def test_list_models_owned_by_anthropic(self):
        with patch("src.providers.anthropic.anthropic.AsyncAnthropic"):
            provider = AnthropicProvider(api_key="sk-ant-test")
        models = await provider.list_models()
        assert all(m.owned_by == "anthropic" for m in models)


class TestAnthropicProviderHealthCheck:
    @pytest.mark.asyncio
    async def test_health_check_returns_healthy_on_success(self):
        mock_resp = _make_mock_anthropic_response()
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_resp)

        with patch("src.providers.anthropic.anthropic.AsyncAnthropic", return_value=mock_client):
            provider = AnthropicProvider(api_key="sk-ant-test")
        provider._client = mock_client

        status = await provider.health_check()
        assert status.healthy is True
        assert status.provider == "anthropic"
        assert status.latency_ms is not None

    @pytest.mark.asyncio
    async def test_health_check_returns_unhealthy_on_failure(self):
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(side_effect=Exception("Connection refused"))

        with patch("src.providers.anthropic.anthropic.AsyncAnthropic", return_value=mock_client):
            provider = AnthropicProvider(api_key="sk-ant-test")
        provider._client = mock_client

        status = await provider.health_check()
        assert status.healthy is False
        assert status.error == "Connection refused"
