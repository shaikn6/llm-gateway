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


class SemanticCache:
    def __init__(self, redis_url: str = "redis://localhost:6379/0", ttl_s: int = 3600):
        self._redis = redis.from_url(redis_url, decode_responses=True)
        self._ttl = ttl_s

    def _key(self, request: dict | list, *, tenant: str) -> str:
        """Key on the whole response-affecting request, scoped to one caller.

        ``request`` must carry everything that can change the answer (provider,
        model, messages, sampling parameters). ``tenant`` is the caller's API
        key; only a fingerprint of it is written to Redis.
        """
        scope = hashlib.sha256(tenant.encode()).hexdigest()[:16]
        payload = json.dumps(request, sort_keys=True)
        return f"llm_cache:{scope}:{hashlib.sha256(payload.encode()).hexdigest()}"

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
