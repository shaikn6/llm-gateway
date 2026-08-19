"""FastAPI LLM gateway application."""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI

from src.api.routes.completions import router as completions_router
from src.api.routes.experiments import router as experiments_router
from src.cache.semantic_cache import SemanticCache
from src.config import settings
from src.gateway.router import GatewayRouter
from src.middleware.usage_tracker import UsageTracker

app = FastAPI(title="LLM Gateway", version="0.1.0")
app.include_router(completions_router)
app.include_router(experiments_router)

@lru_cache(maxsize=1)
def get_router() -> GatewayRouter:
    return GatewayRouter(
        anthropic_key=settings.anthropic_api_key,
        openai_key=settings.openai_api_key,
        ollama_base_url=settings.ollama_base_url,
    )


@lru_cache(maxsize=1)
def get_cache() -> SemanticCache:
    return SemanticCache(redis_url=settings.redis_url, ttl_s=settings.cache_ttl_s)


@lru_cache(maxsize=1)
def get_usage_tracker() -> UsageTracker:
    return UsageTracker(redis_url=settings.redis_url)


@app.get("/health")
def health():
    return {"status": "ok", "version": "0.1.0"}
