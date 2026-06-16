"""FastAPI LLM gateway application."""

from __future__ import annotations

from fastapi import FastAPI

from src.api.routes.completions import router as completions_router
from src.api.routes.experiments import router as experiments_router
from src.config import settings
from src.gateway.router import GatewayRouter

app = FastAPI(title="LLM Gateway", version="0.1.0")
app.include_router(completions_router)
app.include_router(experiments_router)

_gateway: GatewayRouter | None = None


def get_router() -> GatewayRouter:
    global _gateway
    if _gateway is None:
        _gateway = GatewayRouter(
            anthropic_key=settings.anthropic_api_key, openai_key=settings.openai_api_key
        )
    return _gateway


@app.get("/health")
def health():
    return {"status": "ok", "version": "0.1.0"}
