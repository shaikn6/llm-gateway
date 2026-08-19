"""POST /v1/chat/completions — OpenAI-compatible endpoint."""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel

from src.api.deps import require_api_key
from src.config import settings
from src.models.schemas import ChatCompletionRequest, ChatMessage

router = APIRouter(tags=["completions"])


class CompletionRequest(BaseModel):
    model: str = "claude-haiku-4-5"
    messages: list[dict]
    max_tokens: int = 1024


@router.post("/v1/chat/completions")
async def create_completion(
    req: CompletionRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(require_api_key),
):
    from src.api.main import get_cache, get_router, get_usage_tracker

    cache = get_cache() if settings.cache_enabled else None
    if cache is not None:
        cached = cache.get(req.messages)
        if cached is not None:
            return cached

    gateway = get_router()
    provider = gateway.route(req.model)

    chat_request = ChatCompletionRequest(
        model=req.model,
        messages=[ChatMessage(**m) for m in req.messages],
        max_tokens=req.max_tokens,
    )

    try:
        response = await provider.complete(chat_request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    response_dict = response.model_dump(by_alias=True)

    if cache is not None:
        cache.set(req.messages, response_dict)

    background_tasks.add_task(
        get_usage_tracker().record,
        api_key,
        req.model,
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
    )

    return response_dict
