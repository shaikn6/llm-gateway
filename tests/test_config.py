"""Tests for src/config.py — Settings."""

from __future__ import annotations

import pytest
from unittest.mock import patch


class TestSettings:
    def test_default_redis_url(self):
        from src.config import Settings
        s = Settings()
        assert s.redis_url == "redis://localhost:6379/0"

    def test_default_cache_enabled(self):
        from src.config import Settings
        s = Settings()
        assert s.cache_enabled is True

    def test_default_cache_ttl(self):
        from src.config import Settings
        s = Settings()
        assert s.cache_ttl_s == 3600

    def test_default_rate_limit_requests(self):
        from src.config import Settings
        s = Settings()
        assert s.rate_limit_requests == 100

    def test_default_rate_limit_window(self):
        from src.config import Settings
        s = Settings()
        assert s.rate_limit_window_s == 60

    def test_default_api_keys(self):
        from src.config import Settings
        s = Settings()
        assert "dev-key-1" in s.api_keys

    def test_default_log_level(self):
        from src.config import Settings
        s = Settings()
        assert s.log_level == "INFO"

    def test_empty_anthropic_key_default(self):
        from src.config import Settings
        s = Settings()
        assert s.anthropic_api_key == ""

    def test_empty_openai_key_default(self):
        from src.config import Settings
        s = Settings()
        assert s.openai_api_key == ""

    def test_override_via_env(self):
        from src.config import Settings
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test-123"}):
            s = Settings()
        assert s.anthropic_api_key == "sk-test-123"

    def test_cache_enabled_can_be_disabled_via_env(self):
        from src.config import Settings
        with patch.dict("os.environ", {"CACHE_ENABLED": "false"}):
            s = Settings()
        assert s.cache_enabled is False
