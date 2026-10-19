"""Sliding window rate limiter using Redis sorted sets."""

from __future__ import annotations

import time
import uuid

import redis

from src.middleware.audit import key_fingerprint

# Trim, count and (if under the limit) record run as one Lua script. Redis executes a
# script without interleaving other commands, so concurrent callers cannot all read the
# same count and then all be admitted, which is what a separate check + record allowed.
# Returns {allowed, remaining}.
_ACQUIRE_LUA = """
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
redis.call('ZREMRANGEBYSCORE', KEYS[1], 0, now - window)
local count = redis.call('ZCARD', KEYS[1])
if count >= limit then
    return {0, 0}
end
redis.call('ZADD', KEYS[1], now, ARGV[4])
redis.call('EXPIRE', KEYS[1], window * 2)
return {1, limit - count - 1}
"""


class RateLimiter:
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        limit: int = 100,
        window_s: int = 60,
    ):
        self._redis = redis.from_url(redis_url)
        self._limit = limit
        self._window = window_s

    def acquire(self, api_key: str) -> tuple[bool, int]:
        """Atomically take one slot in the key's window.

        Returns (allowed, remaining). A denied call records nothing.
        """
        key = f"ratelimit:{key_fingerprint(api_key)}"
        allowed, remaining = self._redis.eval(
            _ACQUIRE_LUA, 1, key, time.time(), self._window, self._limit, str(uuid.uuid4())
        )
        return bool(allowed), int(remaining)
