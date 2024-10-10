"""Tests for src/gateway/router.py — GatewayRouter."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.gateway.router import GatewayRouter
from src.providers.anthropic import AnthropicProvider


class TestGatewayRouterInit:
    def test_initializes_anthropic_provider(self):
        router = GatewayRouter(anthropic_key="sk-ant-test", openai_key="sk-openai-test")
        assert isinstance(router._anthropic, AnthropicProvider)

    def test_openai_not_initialized_at_startup(self):
        router = GatewayRouter(anthropic_key="sk-ant-test", openai_key="sk-openai-test")
        assert router._openai is None

    def test_openai_key_stored(self):
        router = GatewayRouter(anthropic_key="sk-ant-test", openai_key="my-openai-key")
        assert router._openai_key == "my-openai-key"

    def test_empty_openai_key_is_accepted(self):
        # Should not raise even if openai key is empty
        router = GatewayRouter(anthropic_key="sk-ant-test")
        assert router._openai_key == ""


class TestGatewayRouterRoute:
    def test_claude_prefix_routes_to_anthropic(self):
        router = GatewayRouter(anthropic_key="sk-ant-test")
        provider = router.route("claude-haiku-4-5")
        assert isinstance(provider, AnthropicProvider)

    def test_claude_sonnet_routes_to_anthropic(self):
        router = GatewayRouter(anthropic_key="sk-ant-test")
        provider = router.route("claude-sonnet-4-6")
        assert isinstance(provider, AnthropicProvider)

    def test_gpt_prefix_routes_to_openai(self):
        with patch("src.gateway.router.OpenAIProvider") as MockOpenAI:
            mock_instance = MagicMock()
            MockOpenAI.return_value = mock_instance
            router = GatewayRouter(anthropic_key="sk-ant-test", openai_key="sk-openai-test")
            provider = router.route("gpt-4o")
            assert provider is mock_instance

    def test_gpt4_mini_routes_to_openai(self):
        with patch("src.gateway.router.OpenAIProvider") as MockOpenAI:
            mock_instance = MagicMock()
            MockOpenAI.return_value = mock_instance
            router = GatewayRouter(anthropic_key="sk-ant-test", openai_key="sk-openai-test")
            provider = router.route("gpt-4o-mini")
            assert provider is mock_instance

    def test_o1_prefix_routes_to_openai(self):
        with patch("src.gateway.router.OpenAIProvider") as MockOpenAI:
            mock_instance = MagicMock()
            MockOpenAI.return_value = mock_instance
            router = GatewayRouter(anthropic_key="sk-ant-test", openai_key="sk-openai-test")
            provider = router.route("o1-mini")
            assert provider is mock_instance

    def test_o1_preview_routes_to_openai(self):
        with patch("src.gateway.router.OpenAIProvider") as MockOpenAI:
            mock_instance = MagicMock()
            MockOpenAI.return_value = mock_instance
            router = GatewayRouter(anthropic_key="sk-ant-test", openai_key="sk-openai-test")
            provider = router.route("o1-preview")
            assert provider is mock_instance

    def test_unknown_model_falls_back_to_anthropic(self):
        router = GatewayRouter(anthropic_key="sk-ant-test")
        provider = router.route("some-unknown-model")
        assert isinstance(provider, AnthropicProvider)

    def test_openai_provider_cached_after_first_call(self):
        with patch("src.gateway.router.OpenAIProvider") as MockOpenAI:
            mock_instance = MagicMock()
            MockOpenAI.return_value = mock_instance
            router = GatewayRouter(anthropic_key="sk-ant-test", openai_key="sk-openai-test")
            provider1 = router.route("gpt-4o")
            provider2 = router.route("gpt-4o-mini")
            # Provider class should only be instantiated once
            MockOpenAI.assert_called_once()
            assert provider1 is provider2

    def test_anthropic_provider_returned_on_empty_model(self):
        router = GatewayRouter(anthropic_key="sk-ant-test")
        provider = router.route("")
        assert isinstance(provider, AnthropicProvider)
