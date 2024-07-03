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
        key = sc._key(messages)
        assert key.startswith("llm_cache:")

    def test_key_is_sha256_of_sorted_json(self, cache):
        sc, _ = cache
        messages = [{"role": "user", "content": "hello"}]
        payload = json.dumps(messages, sort_keys=True)
        expected_hash = hashlib.sha256(payload.encode()).hexdigest()
        assert sc._key(messages) == f"llm_cache:{expected_hash}"

    def test_different_messages_produce_different_keys(self, cache):
        sc, _ = cache
        msgs_a = [{"role": "user", "content": "hello"}]
        msgs_b = [{"role": "user", "content": "world"}]
        assert sc._key(msgs_a) != sc._key(msgs_b)

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
        assert sc._key(msgs_a) != sc._key(msgs_b)

    def test_key_sort_keys_true_makes_dict_order_irrelevant(self, cache):
        sc, _ = cache
        msgs_a = [{"content": "hi", "role": "user"}]
        msgs_b = [{"role": "user", "content": "hi"}]
        assert sc._key(msgs_a) == sc._key(msgs_b)


class TestSemanticCacheGet:
    def test_cache_miss_returns_none(self, cache):
        sc, mock_redis = cache
        mock_redis.get.return_value = None
        result = sc.get([{"role": "user", "content": "test"}])
        assert result is None

    def test_cache_hit_returns_parsed_dict(self, cache):
        sc, mock_redis = cache
        stored = {"choices": [{"message": {"content": "cached response"}}]}
        mock_redis.get.return_value = json.dumps(stored)
        result = sc.get([{"role": "user", "content": "test"}])
        assert result == stored

    def test_get_calls_redis_with_correct_key(self, cache):
        sc, mock_redis = cache
        mock_redis.get.return_value = None
        messages = [{"role": "user", "content": "hello"}]
        sc.get(messages)
        expected_key = sc._key(messages)
        mock_redis.get.assert_called_once_with(expected_key)

    def test_cache_hit_returns_dict_not_string(self, cache):
        sc, mock_redis = cache
        mock_redis.get.return_value = json.dumps({"key": "value"})
        result = sc.get([{"role": "user", "content": "test"}])
        assert isinstance(result, dict)


class TestSemanticCacheSet:
    def test_set_calls_setex_with_correct_args(self, cache):
        sc, mock_redis = cache
        messages = [{"role": "user", "content": "hello"}]
        response = {"choices": [{"message": {"content": "response"}}]}
        sc.set(messages, response)
        expected_key = sc._key(messages)
        mock_redis.setex.assert_called_once_with(
            expected_key, 3600, json.dumps(response)
        )

    def test_set_uses_configured_ttl(self):
        mock_redis = MagicMock()
        with patch("src.cache.semantic_cache.redis.from_url", return_value=mock_redis):
            sc = SemanticCache(redis_url="redis://localhost:6379/0", ttl_s=7200)
        messages = [{"role": "user", "content": "hello"}]
        sc.set(messages, {"data": "val"})
        args = mock_redis.setex.call_args[0]
        assert args[1] == 7200

    def test_set_then_get_returns_same_response(self, cache):
        sc, mock_redis = cache
        messages = [{"role": "user", "content": "hello"}]
        response = {"choices": [{"message": {"content": "stored"}}]}
        sc.set(messages, response)
        # Simulate Redis returning the value
        call_args = mock_redis.setex.call_args[0]
        mock_redis.get.return_value = call_args[2]
        result = sc.get(messages)
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
