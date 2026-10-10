"""Tests for src/config.py — Settings."""

from __future__ import annotations

from unittest.mock import patch

import pytest


class TestSettings:
    def test_default_redis_url(self):
        from src.config import Settings
        s = Settings(_env_file=None)
        assert s.redis_url == "redis://localhost:6379/0"

    def test_default_cache_enabled(self):
        from src.config import Settings
        s = Settings(_env_file=None)
        assert s.cache_enabled is True

    def test_default_cache_ttl(self):
        from src.config import Settings
        s = Settings(_env_file=None)
        assert s.cache_ttl_s == 3600

    def test_default_rate_limit_requests(self):
        from src.config import Settings
        s = Settings(_env_file=None)
        assert s.rate_limit_requests == 100

    def test_default_rate_limit_window(self):
        from src.config import Settings
        s = Settings(_env_file=None)
        assert s.rate_limit_window_s == 60

    def test_no_default_api_keys(self):
        from src.config import Settings
        assert Settings.model_fields["api_keys"].default == ""

    def test_default_log_level(self):
        from src.config import Settings
        s = Settings(_env_file=None)
        assert s.log_level == "INFO"

    def test_empty_anthropic_key_default(self):
        from src.config import Settings
        s = Settings(_env_file=None)
        assert s.anthropic_api_key == ""

    def test_empty_openai_key_default(self):
        from src.config import Settings
        s = Settings(_env_file=None)
        assert s.openai_api_key == ""

    def test_override_via_env(self):
        from src.config import Settings
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test-123"}):
            s = Settings(_env_file=None)
        assert s.anthropic_api_key == "sk-test-123"

    def test_cache_enabled_can_be_disabled_via_env(self):
        from src.config import Settings
        with patch.dict("os.environ", {"CACHE_ENABLED": "false"}):
            s = Settings(_env_file=None)
        assert s.cache_enabled is False


class TestApiKeysRequired:
    """The gateway must not come up with a guessable built-in key."""

    def test_refuses_to_start_without_api_keys(self, monkeypatch):
        from src.config import Settings
        monkeypatch.delenv("API_KEYS", raising=False)
        monkeypatch.delenv("GATEWAY_DEV_MODE", raising=False)
        with pytest.raises(ValueError, match="API_KEYS is not set"):
            Settings(_env_file=None)

    def test_blank_api_keys_is_treated_as_unset(self, monkeypatch):
        from src.config import Settings
        monkeypatch.setenv("API_KEYS", " , ,")
        monkeypatch.delenv("GATEWAY_DEV_MODE", raising=False)
        with pytest.raises(ValueError, match="API_KEYS is not set"):
            Settings(_env_file=None)

    def test_configured_keys_are_parsed(self, monkeypatch):
        from src.config import Settings
        monkeypatch.setenv("API_KEYS", "alpha, beta")
        assert Settings(_env_file=None).parsed_api_keys == {"alpha", "beta"}

    def test_dev_mode_allows_start_with_builtin_dev_keys(self, monkeypatch):
        from src.config import DEV_API_KEYS, Settings
        monkeypatch.delenv("API_KEYS", raising=False)
        monkeypatch.setenv("GATEWAY_DEV_MODE", "true")
        assert Settings(_env_file=None).parsed_api_keys == DEV_API_KEYS

    def test_dev_mode_does_not_add_dev_keys_to_configured_keys(self, monkeypatch):
        from src.config import Settings
        monkeypatch.setenv("API_KEYS", "alpha")
        monkeypatch.setenv("GATEWAY_DEV_MODE", "true")
        assert Settings(_env_file=None).parsed_api_keys == {"alpha"}
