"""Tests for src/providers/ollama.py — OllamaProvider."""

from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from src.providers.ollama import OllamaProvider, OLLAMA_DEFAULT_URL
from src.providers.base import ProviderError, ProviderTimeoutError
from src.models.schemas import ChatCompletionRequest, ChatCompletionResponse


def _make_request(**kwargs):
    defaults = dict(
        model="llama3",
        messages=[{"role": "user", "content": "Hello"}],
    )
    defaults.update(kwargs)
    return ChatCompletionRequest(**defaults)


def _make_httpx_response(json_data, status_code=200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    return resp


class TestOllamaProviderInit:
    def test_default_base_url(self):
        with patch("src.providers.ollama.httpx.AsyncClient"):
            provider = OllamaProvider()
        assert provider._base_url == OLLAMA_DEFAULT_URL

    def test_custom_base_url_strips_trailing_slash(self):
        with patch("src.providers.ollama.httpx.AsyncClient"):
            provider = OllamaProvider(base_url="http://myhost:11434/")
        assert provider._base_url == "http://myhost:11434"

    def test_provider_name_is_ollama(self):
        with patch("src.providers.ollama.httpx.AsyncClient"):
            provider = OllamaProvider()
        assert provider.name == "ollama"

    def test_no_api_key_required(self):
        with patch("src.providers.ollama.httpx.AsyncClient"):
            provider = OllamaProvider()
        assert provider.api_key is None


class TestOllamaProviderComplete:
    @pytest.mark.asyncio
    async def test_complete_returns_chat_completion_response(self):
        mock_resp = _make_httpx_response(
            {"message": {"role": "assistant", "content": "Hello from Ollama"}}
        )
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch("src.providers.ollama.httpx.AsyncClient", return_value=mock_client):
            provider = OllamaProvider()
        provider._client = mock_client

        resp = await provider.complete(_make_request())
        assert isinstance(resp, ChatCompletionResponse)
        assert resp.choices[0].message.content == "Hello from Ollama"

    @pytest.mark.asyncio
    async def test_complete_sends_stream_false(self):
        mock_resp = _make_httpx_response({"message": {"content": "ok"}})
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch("src.providers.ollama.httpx.AsyncClient", return_value=mock_client):
            provider = OllamaProvider()
        provider._client = mock_client

        await provider.complete(_make_request())
        call_kwargs = mock_client.post.call_args[1]
        assert call_kwargs["json"]["stream"] is False

    @pytest.mark.asyncio
    async def test_complete_raises_timeout_error(self):
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("timeout"))

        with patch("src.providers.ollama.httpx.AsyncClient", return_value=mock_client):
            provider = OllamaProvider()
        provider._client = mock_client

        with pytest.raises(ProviderTimeoutError):
            await provider.complete(_make_request())

    @pytest.mark.asyncio
    async def test_complete_raises_provider_error_on_http_error(self):
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=httpx.HTTPError("connection error")
        )

        with patch("src.providers.ollama.httpx.AsyncClient", return_value=mock_client):
            provider = OllamaProvider()
        provider._client = mock_client

        with pytest.raises(ProviderError):
            await provider.complete(_make_request())

    @pytest.mark.asyncio
    async def test_complete_sets_model_correctly(self):
        mock_resp = _make_httpx_response({"message": {"content": "ok"}})
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch("src.providers.ollama.httpx.AsyncClient", return_value=mock_client):
            provider = OllamaProvider()
        provider._client = mock_client

        resp = await provider.complete(_make_request(model="mistral"))
        assert resp.model == "mistral"

    @pytest.mark.asyncio
    async def test_complete_id_has_chatcmpl_prefix(self):
        mock_resp = _make_httpx_response({"message": {"content": "ok"}})
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch("src.providers.ollama.httpx.AsyncClient", return_value=mock_client):
            provider = OllamaProvider()
        provider._client = mock_client

        resp = await provider.complete(_make_request())
        assert resp.id.startswith("chatcmpl-")

    @pytest.mark.asyncio
    async def test_complete_finish_reason_is_stop(self):
        mock_resp = _make_httpx_response({"message": {"content": "ok"}})
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch("src.providers.ollama.httpx.AsyncClient", return_value=mock_client):
            provider = OllamaProvider()
        provider._client = mock_client

        resp = await provider.complete(_make_request())
        assert resp.choices[0].finish_reason == "stop"


class TestOllamaProviderListModels:
    @pytest.mark.asyncio
    async def test_list_models_returns_model_objects(self):
        tags_data = {
            "models": [
                {"name": "llama3"},
                {"name": "mistral"},
                {"name": "codellama"},
            ]
        }
        mock_resp = _make_httpx_response(tags_data)
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch("src.providers.ollama.httpx.AsyncClient", return_value=mock_client):
            provider = OllamaProvider()
        provider._client = mock_client

        models = await provider.list_models()
        assert len(models) == 3
        model_ids = {m.id for m in models}
        assert model_ids == {"llama3", "mistral", "codellama"}

    @pytest.mark.asyncio
    async def test_list_models_returns_empty_on_error(self):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=Exception("Connection refused"))

        with patch("src.providers.ollama.httpx.AsyncClient", return_value=mock_client):
            provider = OllamaProvider()
        provider._client = mock_client

        models = await provider.list_models()
        assert models == []

    @pytest.mark.asyncio
    async def test_list_models_owned_by_ollama(self):
        mock_resp = _make_httpx_response({"models": [{"name": "llama3"}]})
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch("src.providers.ollama.httpx.AsyncClient", return_value=mock_client):
            provider = OllamaProvider()
        provider._client = mock_client

        models = await provider.list_models()
        assert models[0].owned_by == "ollama"


class TestOllamaProviderHealthCheck:
    @pytest.mark.asyncio
    async def test_health_check_healthy_on_200(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch("src.providers.ollama.httpx.AsyncClient", return_value=mock_client):
            provider = OllamaProvider()
        provider._client = mock_client

        status = await provider.health_check()
        assert status.healthy is True
        assert status.provider == "ollama"

    @pytest.mark.asyncio
    async def test_health_check_unhealthy_on_non_200(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch("src.providers.ollama.httpx.AsyncClient", return_value=mock_client):
            provider = OllamaProvider()
        provider._client = mock_client

        status = await provider.health_check()
        assert status.healthy is False

    @pytest.mark.asyncio
    async def test_health_check_unhealthy_on_exception(self):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=Exception("refused"))

        with patch("src.providers.ollama.httpx.AsyncClient", return_value=mock_client):
            provider = OllamaProvider()
        provider._client = mock_client

        status = await provider.health_check()
        assert status.healthy is False
        assert "refused" in status.error
