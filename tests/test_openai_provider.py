"""Tests for src/providers/openai.py — OpenAIAsyncProvider."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.providers.openai import OpenAIAsyncProvider, OPENAI_MODELS
from src.providers.base import (
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from src.models.schemas import ChatCompletionRequest, ChatCompletionResponse


def _make_request(**kwargs):
    defaults = dict(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello"}],
    )
    defaults.update(kwargs)
    return ChatCompletionRequest(**defaults)


def _make_mock_openai_response(content="Hello from OpenAI", model="gpt-4o-mini"):
    resp = MagicMock()
    resp.id = "chatcmpl-openai123"
    resp.model = model
    choice = MagicMock()
    choice.message.content = content
    choice.finish_reason = "stop"
    resp.choices = [choice]
    resp.usage = MagicMock(
        prompt_tokens=10, completion_tokens=20, total_tokens=30
    )
    return resp


class TestOpenAIAsyncProviderInit:
    def test_raises_auth_error_when_no_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ProviderAuthError):
                OpenAIAsyncProvider(api_key=None)

    def test_accepts_explicit_api_key(self):
        with patch("src.providers.openai.openai_lib.AsyncOpenAI"):
            provider = OpenAIAsyncProvider(api_key="sk-openai-test")
        assert provider.api_key == "sk-openai-test"

    def test_accepts_env_var_key(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "env-openai-key"}):
            with patch("src.providers.openai.openai_lib.AsyncOpenAI"):
                provider = OpenAIAsyncProvider()
        assert provider.api_key == "env-openai-key"

    def test_provider_name_is_openai(self):
        with patch("src.providers.openai.openai_lib.AsyncOpenAI"):
            provider = OpenAIAsyncProvider(api_key="sk-openai-test")
        assert provider.name == "openai"


class TestOpenAIAsyncProviderComplete:
    @pytest.mark.asyncio
    async def test_complete_returns_chat_completion_response(self):
        mock_resp = _make_mock_openai_response()
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

        with patch("src.providers.openai.openai_lib.AsyncOpenAI", return_value=mock_client):
            provider = OpenAIAsyncProvider(api_key="sk-openai-test")
        provider._client = mock_client

        resp = await provider.complete(_make_request())
        assert isinstance(resp, ChatCompletionResponse)
        assert resp.choices[0].message.content == "Hello from OpenAI"

    @pytest.mark.asyncio
    async def test_complete_maps_usage_correctly(self):
        mock_resp = _make_mock_openai_response()
        mock_resp.usage.prompt_tokens = 50
        mock_resp.usage.completion_tokens = 100
        mock_resp.usage.total_tokens = 150
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

        with patch("src.providers.openai.openai_lib.AsyncOpenAI", return_value=mock_client):
            provider = OpenAIAsyncProvider(api_key="sk-openai-test")
        provider._client = mock_client

        resp = await provider.complete(_make_request())
        assert resp.usage.prompt_tokens == 50
        assert resp.usage.completion_tokens == 100
        assert resp.usage.total_tokens == 150

    @pytest.mark.asyncio
    async def test_complete_raises_auth_error_on_auth_failure(self):
        import openai
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=openai.AuthenticationError(
                message="invalid key",
                response=MagicMock(status_code=401),
                body={},
            )
        )

        with patch("src.providers.openai.openai_lib.AsyncOpenAI", return_value=mock_client):
            provider = OpenAIAsyncProvider(api_key="sk-bad")
        provider._client = mock_client

        with pytest.raises(ProviderAuthError):
            await provider.complete(_make_request())

    @pytest.mark.asyncio
    async def test_complete_raises_rate_limit_error(self):
        import openai
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=openai.RateLimitError(
                message="rate limited",
                response=MagicMock(status_code=429),
                body={},
            )
        )

        with patch("src.providers.openai.openai_lib.AsyncOpenAI", return_value=mock_client):
            provider = OpenAIAsyncProvider(api_key="sk-test")
        provider._client = mock_client

        with pytest.raises(ProviderRateLimitError):
            await provider.complete(_make_request())

    @pytest.mark.asyncio
    async def test_complete_raises_timeout_error(self):
        import openai
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=openai.APITimeoutError(request=MagicMock())
        )

        with patch("src.providers.openai.openai_lib.AsyncOpenAI", return_value=mock_client):
            provider = OpenAIAsyncProvider(api_key="sk-test")
        provider._client = mock_client

        with pytest.raises(ProviderTimeoutError):
            await provider.complete(_make_request())

    @pytest.mark.asyncio
    async def test_complete_raises_provider_error_on_api_error(self):
        import openai
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=openai.APIError(
                message="internal error",
                request=MagicMock(),
                body={},
            )
        )

        with patch("src.providers.openai.openai_lib.AsyncOpenAI", return_value=mock_client):
            provider = OpenAIAsyncProvider(api_key="sk-test")
        provider._client = mock_client

        with pytest.raises(ProviderError):
            await provider.complete(_make_request())

    @pytest.mark.asyncio
    async def test_complete_passes_temperature(self):
        mock_resp = _make_mock_openai_response()
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

        with patch("src.providers.openai.openai_lib.AsyncOpenAI", return_value=mock_client):
            provider = OpenAIAsyncProvider(api_key="sk-test")
        provider._client = mock_client

        await provider.complete(_make_request(temperature=0.7))
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs.get("temperature") == 0.7

    @pytest.mark.asyncio
    async def test_complete_passes_max_tokens(self):
        mock_resp = _make_mock_openai_response()
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

        with patch("src.providers.openai.openai_lib.AsyncOpenAI", return_value=mock_client):
            provider = OpenAIAsyncProvider(api_key="sk-test")
        provider._client = mock_client

        await provider.complete(_make_request(max_tokens=512))
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs.get("max_tokens") == 512


class TestOpenAIAsyncProviderListModels:
    @pytest.mark.asyncio
    async def test_list_models_returns_all_openai_models(self):
        with patch("src.providers.openai.openai_lib.AsyncOpenAI"):
            provider = OpenAIAsyncProvider(api_key="sk-test")
        models = await provider.list_models()
        model_ids = {m.id for m in models}
        assert model_ids == set(OPENAI_MODELS.keys())

    @pytest.mark.asyncio
    async def test_list_models_owned_by_openai(self):
        with patch("src.providers.openai.openai_lib.AsyncOpenAI"):
            provider = OpenAIAsyncProvider(api_key="sk-test")
        models = await provider.list_models()
        assert all(m.owned_by == "openai" for m in models)


class TestOpenAIAsyncProviderHealthCheck:
    @pytest.mark.asyncio
    async def test_health_check_healthy_on_success(self):
        mock_client = AsyncMock()
        mock_client.models.list = AsyncMock(return_value=MagicMock())

        with patch("src.providers.openai.openai_lib.AsyncOpenAI", return_value=mock_client):
            provider = OpenAIAsyncProvider(api_key="sk-test")
        provider._client = mock_client

        status = await provider.health_check()
        assert status.healthy is True
        assert status.provider == "openai"

    @pytest.mark.asyncio
    async def test_health_check_unhealthy_on_exception(self):
        mock_client = AsyncMock()
        mock_client.models.list = AsyncMock(side_effect=Exception("network error"))

        with patch("src.providers.openai.openai_lib.AsyncOpenAI", return_value=mock_client):
            provider = OpenAIAsyncProvider(api_key="sk-test")
        provider._client = mock_client

        status = await provider.health_check()
        assert status.healthy is False
        assert "network error" in status.error
