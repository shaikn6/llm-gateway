"""The gateway must bound the Redis socket so an unreachable-but-not-refusing
Redis fails fast instead of hanging the request. Every Redis client passes the
configured timeout to both socket_connect_timeout and socket_timeout.

These assertions are deterministic (they inspect the kwargs handed to
redis.from_url) so they stay CI-safe; a real black-hole connection test would
be flaky.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.cache.semantic_cache import SemanticCache
from src.middleware.rate_limiter import RateLimiter
from src.middleware.usage_tracker import UsageTracker


def _timeout_kwargs(from_url: MagicMock) -> dict:
    return from_url.call_args.kwargs


class TestRedisTimeoutPassedToFromUrl:
    def test_rate_limiter_bounds_socket(self):
        with patch("src.middleware.rate_limiter.redis.from_url") as from_url:
            RateLimiter(timeout_s=1.5)
        kwargs = _timeout_kwargs(from_url)
        assert kwargs["socket_connect_timeout"] == 1.5
        assert kwargs["socket_timeout"] == 1.5

    def test_usage_tracker_bounds_socket(self):
        with patch("src.middleware.usage_tracker.redis.from_url") as from_url:
            UsageTracker(timeout_s=1.5)
        kwargs = _timeout_kwargs(from_url)
        assert kwargs["socket_connect_timeout"] == 1.5
        assert kwargs["socket_timeout"] == 1.5

    def test_semantic_cache_bounds_socket(self):
        with patch("src.cache.semantic_cache.redis.from_url") as from_url:
            SemanticCache(timeout_s=1.5)
        kwargs = _timeout_kwargs(from_url)
        assert kwargs["socket_connect_timeout"] == 1.5
        assert kwargs["socket_timeout"] == 1.5

    def test_default_timeout_is_bounded(self):
        """A client built with no explicit timeout is still bounded, not unlimited."""
        with patch("src.middleware.rate_limiter.redis.from_url") as from_url:
            RateLimiter()
        kwargs = _timeout_kwargs(from_url)
        assert kwargs["socket_connect_timeout"] == 2.0
        assert kwargs["socket_timeout"] == 2.0
