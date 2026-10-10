"""Tests for src/cache/semantic_cache.py."""

from __future__ import annotations

import hashlib
import json
from unittest.mock import MagicMock, patch

import pytest

from src.cache.semantic_cache import SemanticCache


@pytest.fixture
def mock_redis():
    """Return a mock Redis client."""
    return MagicMock()


@pytest.fixture
def cache(mock_redis):
    """Return a SemanticCache instance with a mocked Redis client."""
    with patch("src.cache.semantic_cache.redis.from_url", return_value=mock_redis):
        sc = SemanticCache(redis_url="redis://localhost:6379/0", ttl_s=3600)
    return sc, mock_redis


class TestSemanticCacheKey:
    def test_key_has_llm_cache_prefix(self, cache):
        sc, _ = cache
        messages = [{"role": "user", "content": "hello"}]
        key = sc._key(messages, tenant="key-a")
        assert key.startswith("llm_cache:")

    def test_key_is_tenant_fingerprint_plus_sha256_of_sorted_json(self, cache):
        sc, _ = cache
        messages = [{"role": "user", "content": "hello"}]
        payload = json.dumps(messages, sort_keys=True)
        expected_hash = hashlib.sha256(payload.encode()).hexdigest()
        tenant = hashlib.sha256(b"key-a").hexdigest()[:16]
        assert sc._key(messages, tenant="key-a") == f"llm_cache:{tenant}:{expected_hash}"

    def test_raw_api_key_never_appears_in_the_redis_key(self, cache):
        sc, _ = cache
        assert "key-a" not in sc._key([{"role": "user", "content": "hello"}], tenant="key-a")

    def test_different_tenants_produce_different_keys(self, cache):
        sc, _ = cache
        messages = [{"role": "user", "content": "hello"}]
        assert sc._key(messages, tenant="key-a") != sc._key(messages, tenant="key-b")

    def test_tenant_is_required(self, cache):
        sc, _ = cache
        with pytest.raises(TypeError):
            sc.get([{"role": "user", "content": "hello"}])

    def test_any_request_field_changes_the_key(self, cache):
        sc, _ = cache
        base = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "temperature": None}
        assert sc._key(base, tenant="key-a") != sc._key({**base, "model": "n"}, tenant="key-a")
        assert sc._key(base, tenant="key-a") != sc._key({**base, "temperature": 0.2}, tenant="key-a")

    def test_different_messages_produce_different_keys(self, cache):
        sc, _ = cache
        msgs_a = [{"role": "user", "content": "hello"}]
        msgs_b = [{"role": "user", "content": "world"}]
        assert sc._key(msgs_a, tenant="key-a") != sc._key(msgs_b, tenant="key-a")

    def test_message_order_affects_key(self, cache):
        sc, _ = cache
        msgs_a = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
        ]
        msgs_b = [
            {"role": "assistant", "content": "second"},
            {"role": "user", "content": "first"},
        ]
        assert sc._key(msgs_a, tenant="key-a") != sc._key(msgs_b, tenant="key-a")

    def test_key_sort_keys_true_makes_dict_order_irrelevant(self, cache):
        sc, _ = cache
        msgs_a = [{"content": "hi", "role": "user"}]
        msgs_b = [{"role": "user", "content": "hi"}]
        assert sc._key(msgs_a, tenant="key-a") == sc._key(msgs_b, tenant="key-a")


class TestSemanticCacheGet:
    def test_cache_miss_returns_none(self, cache):
        sc, mock_redis = cache
        mock_redis.get.return_value = None
        result = sc.get([{"role": "user", "content": "test"}], tenant="key-a")
        assert result is None

    def test_cache_hit_returns_parsed_dict(self, cache):
        sc, mock_redis = cache
        stored = {"choices": [{"message": {"content": "cached response"}}]}
        mock_redis.get.return_value = json.dumps(stored)
        result = sc.get([{"role": "user", "content": "test"}], tenant="key-a")
        assert result == stored

    def test_get_calls_redis_with_correct_key(self, cache):
        sc, mock_redis = cache
        mock_redis.get.return_value = None
        messages = [{"role": "user", "content": "hello"}]
        sc.get(messages, tenant="key-a")
        expected_key = sc._key(messages, tenant="key-a")
        mock_redis.get.assert_called_once_with(expected_key)

    def test_cache_hit_returns_dict_not_string(self, cache):
        sc, mock_redis = cache
        mock_redis.get.return_value = json.dumps({"key": "value"})
        result = sc.get([{"role": "user", "content": "test"}], tenant="key-a")
        assert isinstance(result, dict)


class TestSemanticCacheSet:
    def test_set_calls_setex_with_correct_args(self, cache):
        sc, mock_redis = cache
        messages = [{"role": "user", "content": "hello"}]
        response = {"choices": [{"message": {"content": "response"}}]}
        sc.set(messages, response, tenant="key-a")
        expected_key = sc._key(messages, tenant="key-a")
        mock_redis.setex.assert_called_once_with(
            expected_key, 3600, json.dumps(response)
        )

    def test_set_uses_configured_ttl(self):
        mock_redis = MagicMock()
        with patch("src.cache.semantic_cache.redis.from_url", return_value=mock_redis):
            sc = SemanticCache(redis_url="redis://localhost:6379/0", ttl_s=7200)
        messages = [{"role": "user", "content": "hello"}]
        sc.set(messages, {"data": "val"}, tenant="key-a")
        args = mock_redis.setex.call_args[0]
        assert args[1] == 7200

    def test_set_then_get_returns_same_response(self, cache):
        sc, mock_redis = cache
        messages = [{"role": "user", "content": "hello"}]
        response = {"choices": [{"message": {"content": "stored"}}]}
        sc.set(messages, response, tenant="key-a")
        # Simulate Redis returning the value
        call_args = mock_redis.setex.call_args[0]
        mock_redis.get.return_value = call_args[2]
        result = sc.get(messages, tenant="key-a")
        assert result == response


class TestSemanticCacheInit:
    def test_default_ttl_is_3600(self):
        mock_redis = MagicMock()
        with patch("src.cache.semantic_cache.redis.from_url", return_value=mock_redis):
            sc = SemanticCache()
        assert sc._ttl == 3600

    def test_custom_ttl_is_stored(self):
        mock_redis = MagicMock()
        with patch("src.cache.semantic_cache.redis.from_url", return_value=mock_redis):
            sc = SemanticCache(ttl_s=1800)
        assert sc._ttl == 1800

    def test_redis_url_passed_to_from_url(self):
        with patch("src.cache.semantic_cache.redis.from_url") as mock_from_url:
            mock_from_url.return_value = MagicMock()
            SemanticCache(redis_url="redis://myhost:6380/1")
        mock_from_url.assert_called_once_with(
            "redis://myhost:6380/1", decode_responses=True
        )


class TestSemanticCacheRedisOutage:
    """A Redis failure must look like a cache miss, never an error."""

    def test_get_degrades_to_miss_when_redis_is_down(self, cache):
        import redis

        sc, mock_redis = cache
        mock_redis.get.side_effect = redis.ConnectionError("down")
        assert sc.get([{"role": "user", "content": "test"}], tenant="key-a") is None

    def test_set_is_a_noop_when_redis_is_down(self, cache):
        import redis

        sc, mock_redis = cache
        mock_redis.setex.side_effect = redis.TimeoutError("slow")
        sc.set([{"role": "user", "content": "test"}], {"ok": True}, tenant="key-a")
