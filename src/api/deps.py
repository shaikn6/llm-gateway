"""Shared FastAPI dependencies: API-key auth and rate limiting."""

from __future__ import annotations

from functools import lru_cache

from fastapi import Depends, Header, HTTPException

from src.config import settings
from src.middleware.rate_limiter import RateLimiter


@lru_cache(maxsize=1)
def get_rate_limiter() -> RateLimiter:
    return RateLimiter(
        redis_url=settings.redis_url,
        limit=settings.rate_limit_requests,
        window_s=settings.rate_limit_window_s,
    )


def get_api_key(x_api_key: str = Header(default="")) -> str:
    """Validate X-API-Key against the configured keys. Raises 401 if missing/unknown."""
    if not x_api_key or x_api_key not in settings.parsed_api_keys:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")
    return x_api_key


def require_api_key(api_key: str = Depends(get_api_key)) -> str:
    """Auth (via get_api_key) plus sliding-window rate limiting.

    Raises 429 once the caller's quota is exhausted. Returns the validated
    key so route handlers can use it for caching/usage-tracking without
    re-parsing the header.
    """
    limiter = get_rate_limiter()
    allowed, _remaining = limiter.check(api_key)
    if not allowed:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    limiter.record(api_key)
    return api_key
