"""Exact-match Redis cache for LLM responses.

Despite the historical class name this is not a semantic (embedding) cache:
a hit requires a byte-identical request from the same caller.
"""

from __future__ import annotations

import hashlib
import json
import logging

import redis

logger = logging.getLogger(__name__)


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class SemanticCache:
    def __init__(
        self, redis_url: str = "redis://localhost:6379/0", ttl_s: int = 3600, timeout_s: float = 2.0
    ):
        self._redis = redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=timeout_s,
            socket_timeout=timeout_s,
        )
        self._ttl = ttl_s

    def _key(self, request: dict | list, *, tenant: str) -> str:
        """Key on the whole response-affecting request, scoped to one caller.

        ``request`` must carry everything that can change the answer (provider,
        model, messages, sampling parameters). ``tenant`` is the caller's API
        key; only a fingerprint of it is written to Redis.
        """
        return f"llm_cache:{self._scope(tenant)}:{digest(request)}"

    @staticmethod
    def _scope(tenant: str) -> str:
        return hashlib.sha256(tenant.encode()).hexdigest()[:16]

    def get(self, request: dict | list, *, tenant: str) -> dict | None:
        try:
            val = self._redis.get(self._key(request, tenant=tenant))
        except redis.RedisError as exc:
            # The cache is an optimisation: an outage is a miss, not an error.
            logger.warning("cache read failed (%s); treating as a miss", type(exc).__name__)
            return None
        return json.loads(val) if val else None

    def set(self, request: dict | list, response: dict, *, tenant: str) -> None:
        try:
            self._redis.setex(self._key(request, tenant=tenant), self._ttl, json.dumps(response))
        except redis.RedisError as exc:
            logger.warning("cache write failed (%s); response not cached", type(exc).__name__)

    def clear(self, *, tenant: str) -> int:
        """Invalidate only this caller's entries, using incremental Redis SCAN."""
        deleted = 0
        try:
            cursor = 0
            while True:
                cursor, batch = self._redis.scan(
                    cursor=cursor, match=f"llm_cache:{self._scope(tenant)}:*", count=100
                )
                if batch:
                    deleted += self._redis.unlink(*batch)
                if not cursor:
                    break
        except redis.RedisError as exc:
            logger.warning("cache invalidation failed (%s)", type(exc).__name__)
        return deleted
