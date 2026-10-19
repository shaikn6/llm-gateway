"""Concurrency regression tests for the rate limiter.

Runs against fakeredis (with Lua) instead of a MagicMock: the bug these guard against
only exists when commands from different callers can interleave on a shared store.
"""

from __future__ import annotations

import threading
from unittest.mock import patch

import fakeredis
import pytest
from fastapi import HTTPException

from src.api import deps as deps_module
from src.middleware.audit import key_fingerprint
from src.middleware.rate_limiter import RateLimiter

LIMIT = 10
CALLERS = 64


@pytest.fixture
def server():
    return fakeredis.FakeServer()


@pytest.fixture
def limiter(server):
    with patch(
        "src.middleware.rate_limiter.redis.from_url",
        side_effect=lambda *_a, **_k: fakeredis.FakeRedis(server=server),
    ):
        deps_module.get_rate_limiter.cache_clear()
        try:
            yield RateLimiter(limit=LIMIT, window_s=60)
        finally:
            deps_module.get_rate_limiter.cache_clear()


def _hammer(call, callers: int = CALLERS) -> list[bool]:
    """Fire `call` from `callers` threads released together; return who was admitted."""
    barrier = threading.Barrier(callers)
    results: list[bool] = [False] * callers

    def worker(i: int) -> None:
        barrier.wait()
        results[i] = call()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(callers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def test_dependency_never_admits_more_than_the_limit_under_concurrency(server):
    """Through the real auth dependency: 64 simultaneous callers, limit 10."""

    def admitted() -> bool:
        try:
            deps_module.require_api_key("dev-key-1")
        except HTTPException as exc:
            assert exc.status_code == 429
            return False
        return True

    with (
        patch(
            "src.middleware.rate_limiter.redis.from_url",
            side_effect=lambda *_a, **_k: fakeredis.FakeRedis(server=server),
        ),
        patch.object(deps_module.settings, "rate_limit_requests", LIMIT),
    ):
        deps_module.get_rate_limiter.cache_clear()
        try:
            deps_module.get_rate_limiter()  # build the singleton before the threads start
            results = _hammer(admitted)
        finally:
            deps_module.get_rate_limiter.cache_clear()

    assert sum(results) == LIMIT


def test_acquire_admits_exactly_the_limit_under_concurrency(limiter, server):
    results = _hammer(lambda: limiter.acquire("burst-key")[0])

    assert sum(results) == LIMIT
    stored = fakeredis.FakeRedis(server=server).zcard(f"ratelimit:{key_fingerprint('burst-key')}")
    assert stored == LIMIT  # denied callers recorded nothing
