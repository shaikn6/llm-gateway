"""Tests for src/middleware/rate_limiter.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import fakeredis
import pytest
import redis

from src.middleware.audit import key_fingerprint
from src.middleware.rate_limiter import RateLimiter


@pytest.fixture
def server():
    return fakeredis.FakeServer()


@pytest.fixture
def make_limiter(server):
    def _make(limit: int = 3, window_s: int = 60) -> RateLimiter:
        with patch(
            "src.middleware.rate_limiter.redis.from_url",
            return_value=fakeredis.FakeRedis(server=server),
        ):
            return RateLimiter(limit=limit, window_s=window_s)

    return _make


class TestRateLimiterAcquire:
    def test_admits_until_limit_then_denies(self, make_limiter):
        rl = make_limiter(limit=3)

        assert [rl.acquire("k") for _ in range(4)] == [
            (True, 2),
            (True, 1),
            (True, 0),
            (False, 0),
        ]

    def test_denied_call_records_nothing(self, make_limiter, server):
        rl = make_limiter(limit=1)
        rl.acquire("k")
        rl.acquire("k")
        rl.acquire("k")

        zset = f"ratelimit:{key_fingerprint('k')}"
        assert fakeredis.FakeRedis(server=server).zcard(zset) == 1

    def test_keys_are_independent(self, make_limiter):
        rl = make_limiter(limit=1)

        assert rl.acquire("a")[0] is True
        assert rl.acquire("b")[0] is True
        assert rl.acquire("a")[0] is False

    def test_stores_under_fingerprint_not_raw_key(self, make_limiter, server):
        rl = make_limiter()
        rl.acquire("secret-key")

        keys = fakeredis.FakeRedis(server=server).keys("*")
        assert keys == [f"ratelimit:{key_fingerprint('secret-key')}".encode()]

    def test_window_slides(self, make_limiter):
        rl = make_limiter(limit=1, window_s=60)
        with patch("src.middleware.rate_limiter.time.time", return_value=1000.0):
            assert rl.acquire("k")[0] is True
        with patch("src.middleware.rate_limiter.time.time", return_value=1059.0):
            assert rl.acquire("k")[0] is False  # first request still inside the window
        with patch("src.middleware.rate_limiter.time.time", return_value=1061.0):
            assert rl.acquire("k")[0] is True  # first request has aged out

    def test_sets_ttl_of_double_the_window(self, make_limiter, server):
        rl = make_limiter(window_s=30)
        rl.acquire("k")

        zset = f"ratelimit:{key_fingerprint('k')}"
        assert fakeredis.FakeRedis(server=server).ttl(zset) == 60

    def test_redis_error_propagates(self):
        mock_redis = MagicMock()
        mock_redis.eval.side_effect = redis.ConnectionError("down")
        with patch("src.middleware.rate_limiter.redis.from_url", return_value=mock_redis):
            rl = RateLimiter()

        with pytest.raises(redis.ConnectionError):
            rl.acquire("k")


class TestRateLimiterInit:
    def test_default_limit_is_100(self):
        mock_redis = MagicMock()
        with patch("src.middleware.rate_limiter.redis.from_url", return_value=mock_redis):
            rl = RateLimiter()
        assert rl._limit == 100

    def test_default_window_is_60(self):
        mock_redis = MagicMock()
        with patch("src.middleware.rate_limiter.redis.from_url", return_value=mock_redis):
            rl = RateLimiter()
        assert rl._window == 60

    def test_custom_limit_and_window(self):
        mock_redis = MagicMock()
        with patch("src.middleware.rate_limiter.redis.from_url", return_value=mock_redis):
            rl = RateLimiter(limit=50, window_s=30)
        assert rl._limit == 50
        assert rl._window == 30
