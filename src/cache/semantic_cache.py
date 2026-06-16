"""Hash-based Redis cache for LLM responses."""

from __future__ import annotations

import hashlib
import json

import redis


class SemanticCache:
    def __init__(self, redis_url: str = "redis://localhost:6379/0", ttl_s: int = 3600):
        self._redis = redis.from_url(redis_url, decode_responses=True)
        self._ttl = ttl_s

    def _key(self, messages: list[dict]) -> str:
        payload = json.dumps(messages, sort_keys=True)
        return f"llm_cache:{hashlib.sha256(payload.encode()).hexdigest()}"

    def get(self, messages: list[dict]) -> dict | None:
        val = self._redis.get(self._key(messages))
        return json.loads(val) if val else None

    def set(self, messages: list[dict], response: dict) -> None:
        self._redis.setex(self._key(messages), self._ttl, json.dumps(response))
