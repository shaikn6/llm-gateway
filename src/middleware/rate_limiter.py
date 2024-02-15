"""Sliding window rate limiter using Redis sorted sets."""

from __future__ import annotations

import time
import uuid

import redis


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

    def check(self, api_key: str) -> tuple[bool, int]:
        key = f"ratelimit:{api_key}"
        now = time.time()
        pipe = self._redis.pipeline()
        pipe.zremrangebyscore(key, 0, now - self._window)
        pipe.zcard(key)
        _, count = pipe.execute()
        allowed = count < self._limit
        return allowed, max(0, self._limit - count - 1)

    def record(self, api_key: str) -> None:
        key = f"ratelimit:{api_key}"
        now = time.time()
        self._redis.zadd(key, {str(uuid.uuid4()): now})
        self._redis.expire(key, self._window * 2)
