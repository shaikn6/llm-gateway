"""Tests for src/middleware/rate_limiter.py."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from src.middleware.rate_limiter import RateLimiter


@pytest.fixture
def mock_redis():
    return MagicMock()


@pytest.fixture
def limiter(mock_redis):
    with patch("src.middleware.rate_limiter.redis.from_url", return_value=mock_redis):
        rl = RateLimiter(redis_url="redis://localhost:6379/0", limit=100, window_s=60)
    return rl, mock_redis


class TestRateLimiterCheck:
    def test_allowed_when_under_limit(self, limiter):
        rl, mock_redis = limiter
        # pipeline returns: (None, 50) — 50 requests in window
        pipe = MagicMock()
        pipe.execute.return_value = [None, 50]
        mock_redis.pipeline.return_value = pipe

        allowed, remaining = rl.check("api-key-1")

        assert allowed is True
        assert remaining == 49  # 100 - 50 - 1

    def test_denied_when_at_limit(self, limiter):
        rl, mock_redis = limiter
        pipe = MagicMock()
        pipe.execute.return_value = [None, 100]
        mock_redis.pipeline.return_value = pipe

        allowed, remaining = rl.check("api-key-1")

        assert allowed is False
        assert remaining == 0

    def test_denied_when_over_limit(self, limiter):
        rl, mock_redis = limiter
        pipe = MagicMock()
        pipe.execute.return_value = [None, 150]
        mock_redis.pipeline.return_value = pipe

        allowed, remaining = rl.check("api-key-1")

        assert allowed is False
        assert remaining == 0  # max(0, 100 - 150 - 1) => 0

    def test_remaining_is_zero_when_one_under_limit(self, limiter):
        rl, mock_redis = limiter
        pipe = MagicMock()
        pipe.execute.return_value = [None, 99]
        mock_redis.pipeline.return_value = pipe

        allowed, remaining = rl.check("api-key-1")

        assert allowed is True
        assert remaining == 0

    def test_check_uses_correct_redis_key(self, limiter):
        rl, mock_redis = limiter
        pipe = MagicMock()
        pipe.execute.return_value = [None, 0]
        mock_redis.pipeline.return_value = pipe

        rl.check("my-api-key")

        pipe.zremrangebyscore.assert_called_once()
        args = pipe.zremrangebyscore.call_args[0]
        assert args[0] == "ratelimit:my-api-key"

    def test_check_removes_stale_entries_before_counting(self, limiter):
        rl, mock_redis = limiter
        pipe = MagicMock()
        pipe.execute.return_value = [None, 5]
        mock_redis.pipeline.return_value = pipe

        rl.check("api-key-1")

        # zremrangebyscore should be called to remove expired entries
        pipe.zremrangebyscore.assert_called_once()
        pipe.zcard.assert_called_once_with("ratelimit:api-key-1")

    def test_empty_window_returns_max_remaining(self, limiter):
        rl, mock_redis = limiter
        pipe = MagicMock()
        pipe.execute.return_value = [None, 0]
        mock_redis.pipeline.return_value = pipe

        allowed, remaining = rl.check("api-key-1")

        assert allowed is True
        assert remaining == 99  # 100 - 0 - 1


class TestRateLimiterRecord:
    def test_record_adds_to_sorted_set(self, limiter):
        rl, mock_redis = limiter
        rl.record("api-key-1")
        mock_redis.zadd.assert_called_once()
        call_args = mock_redis.zadd.call_args[0]
        assert call_args[0] == "ratelimit:api-key-1"

    def test_record_sets_expire(self, limiter):
        rl, mock_redis = limiter
        rl.record("api-key-1")
        mock_redis.expire.assert_called_once_with("ratelimit:api-key-1", 120)  # window_s * 2

    def test_record_uses_current_timestamp_as_score(self, limiter):
        rl, mock_redis = limiter
        before = time.time()
        rl.record("api-key-1")
        after = time.time()

        call_args = mock_redis.zadd.call_args[0]
        score = list(call_args[1].values())[0]
        assert before <= score <= after

    def test_record_uses_unique_uuid_as_member(self, limiter):
        rl, mock_redis = limiter
        rl.record("key")
        rl.record("key")
        all_members = [
            list(call[0][1].keys())[0]
            for call in mock_redis.zadd.call_args_list
        ]
        assert all_members[0] != all_members[1]


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
