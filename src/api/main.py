"""FastAPI LLM gateway application."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

from src import __version__
from src.api.routes.completions import router as completions_router
from src.api.routes.experiments import router as experiments_router
from src.api.routes.usage import router as usage_router
from src.cache.semantic_cache import SemanticCache
from src.config import settings
from src.gateway.router import GatewayRouter
from src.middleware.audit import AuditMiddleware, configure_logging
from src.middleware.body_limit import MaxBodySizeMiddleware
from src.middleware.metrics import metrics
from src.middleware.usage_tracker import UsageTracker

configure_logging()


@asynccontextmanager
async def lifespan(app):
    if settings.semantic_cache_enabled:
        cache = await asyncio.to_thread(get_embedding_cache)
        await asyncio.to_thread(cache.warm)
    yield


app = FastAPI(title="LLM Gateway", version=__version__, lifespan=lifespan)
app.add_middleware(AuditMiddleware)
app.add_middleware(MaxBodySizeMiddleware, max_bytes=settings.max_body_bytes)
app.include_router(completions_router)
app.include_router(experiments_router)
app.include_router(usage_router)


@lru_cache(maxsize=1)
def get_router() -> GatewayRouter:
    return GatewayRouter(
        anthropic_key=settings.anthropic_api_key,
        openai_key=settings.openai_api_key,
        ollama_base_url=settings.ollama_base_url,
    )


@lru_cache(maxsize=1)
def get_cache() -> SemanticCache:
    return SemanticCache(
        redis_url=settings.redis_url, ttl_s=settings.cache_ttl_s, timeout_s=settings.redis_timeout_s
    )


@lru_cache(maxsize=1)
def get_usage_tracker() -> UsageTracker:
    return UsageTracker(redis_url=settings.redis_url, timeout_s=settings.redis_timeout_s)


@app.get("/health")
def health():
    return {"status": "ok", "version": __version__}


@app.get("/metrics", include_in_schema=False)
def prometheus_metrics():
    # Unauthenticated and outside the rate limiter, like /health, so a scraper
    # can poll it without a key and without consuming anyone's quota.
    return PlainTextResponse(metrics.render(), media_type="text/plain; version=0.0.4")


@lru_cache(maxsize=1)
def get_embedding_cache():
    try:
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        from src.cache.embedding_cache import ChromaStore, EmbeddingCache

        embedder = DefaultEmbeddingFunction()
        return EmbeddingCache(
            embedder,
            ChromaStore(settings.semantic_cache_path),
            settings.semantic_cache_threshold,
            settings.cache_ttl_s,
            settings.semantic_cache_max_entries_per_tenant,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Semantic cache startup failed: install .[semantic] and ensure the default "
            "Chroma embedder is available locally."
        ) from exc
