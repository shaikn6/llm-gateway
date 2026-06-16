"""Tests for src/providers/base.py — LLMProvider abstract base and exceptions."""

from __future__ import annotations

import pytest

from src.providers.base import (
    LLMProvider,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)


class TestProviderError:
    def test_base_error_defaults(self):
        err = ProviderError("something failed")
        assert str(err) == "something failed"
        assert err.status_code == 500
        assert err.provider == "unknown"

    def test_custom_status_code(self):
        err = ProviderError("not found", status_code=404, provider="openai")
        assert err.status_code == 404
        assert err.provider == "openai"

    def test_is_exception_subclass(self):
        assert issubclass(ProviderError, Exception)


class TestProviderAuthError:
    def test_status_code_is_401(self):
        err = ProviderAuthError("bad key")
        assert err.status_code == 401

    def test_is_provider_error_subclass(self):
        assert issubclass(ProviderAuthError, ProviderError)

    def test_provider_stored(self):
        err = ProviderAuthError("bad key", provider="anthropic")
        assert err.provider == "anthropic"


class TestProviderRateLimitError:
    def test_status_code_is_429(self):
        err = ProviderRateLimitError("rate limited")
        assert err.status_code == 429

    def test_retry_after_defaults_to_none(self):
        err = ProviderRateLimitError("rate limited")
        assert err.retry_after is None

    def test_retry_after_can_be_set(self):
        err = ProviderRateLimitError("rate limited", retry_after=30.0)
        assert err.retry_after == 30.0

    def test_is_provider_error_subclass(self):
        assert issubclass(ProviderRateLimitError, ProviderError)


class TestProviderTimeoutError:
    def test_status_code_is_504(self):
        err = ProviderTimeoutError("timed out")
        assert err.status_code == 504

    def test_is_provider_error_subclass(self):
        assert issubclass(ProviderTimeoutError, ProviderError)


class TestLLMProviderAbstract:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            LLMProvider()

    def test_concrete_subclass_must_implement_complete(self):
        class BadProvider(LLMProvider):
            name = "bad"
            async def stream(self, req): ...
            async def list_models(self): ...
            async def health_check(self): ...

        with pytest.raises(TypeError):
            BadProvider()

    def test_count_tokens_estimate_minimum_is_one(self):
        class ConcreteProvider(LLMProvider):
            name = "test"
            async def complete(self, req): ...
            async def stream(self, req): ...
            async def list_models(self): ...
            async def health_check(self): ...

        p = ConcreteProvider(api_key="test")
        assert p._count_tokens_estimate("") == 1
        assert p._count_tokens_estimate("a") == 1

    def test_count_tokens_estimate_for_longer_text(self):
        class ConcreteProvider(LLMProvider):
            name = "test"
            async def complete(self, req): ...
            async def stream(self, req): ...
            async def list_models(self): ...
            async def health_check(self): ...

        p = ConcreteProvider(api_key="test")
        # 400 chars / 4 = 100 tokens
        text = "a" * 400
        assert p._count_tokens_estimate(text) == 100

    def test_init_stores_api_key_and_timeout(self):
        class ConcreteProvider(LLMProvider):
            name = "test"
            async def complete(self, req): ...
            async def stream(self, req): ...
            async def list_models(self): ...
            async def health_check(self): ...

        p = ConcreteProvider(api_key="mykey", timeout=30.0)
        assert p.api_key == "mykey"
        assert p.timeout == 30.0

    def test_healthy_defaults_to_true(self):
        class ConcreteProvider(LLMProvider):
            name = "test"
            async def complete(self, req): ...
            async def stream(self, req): ...
            async def list_models(self): ...
            async def health_check(self): ...

        p = ConcreteProvider(api_key="test")
        assert p._healthy is True

    @pytest.mark.asyncio
    async def test_is_healthy_calls_health_check_when_stale(self):
        from src.models.schemas import HealthStatus

        class ConcreteProvider(LLMProvider):
            name = "test"
            async def complete(self, req): ...
            async def stream(self, req): ...
            async def list_models(self): ...
            async def health_check(self):
                return HealthStatus(provider="test", healthy=True)

        p = ConcreteProvider(api_key="test")
        # Force stale check
        p._last_health_check = 0
        result = await p.is_healthy()
        assert result is True
