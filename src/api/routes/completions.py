"""POST /v1/chat/completions — OpenAI-compatible endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

router = APIRouter(tags=["completions"])


class CompletionRequest(BaseModel):
    model: str = "claude-haiku-4-5"
    messages: list[dict]
    max_tokens: int = 1024


@router.post("/v1/chat/completions")
def create_completion(req: CompletionRequest, x_api_key: str = Header(default="")):
    from src.api.main import get_router

    gateway = get_router()
    provider = gateway.route(req.model)
    try:
        return provider.complete(req.messages, model=req.model, max_tokens=req.max_tokens)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
