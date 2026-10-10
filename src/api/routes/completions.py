"""POST /v1/chat/completions — OpenAI-compatible endpoint."""

from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator, model_validator

from src.api.deps import require_api_key
from src.config import settings
from src.models.schemas import ChatCompletionRequest, ChatMessage
from src.providers.base import ProviderError, ProviderTimeoutError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["completions"])


class CompletionRequest(BaseModel):
    """The subset of the OpenAI chat-completions request this gateway honours.

    Anything else is rejected rather than accepted and silently dropped.
    """

    model: str = "claude-haiku-4-5"
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int = Field(default=1024, ge=1)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    stop: str | list[str] | None = None
    stream: bool = False

    @model_validator(mode="before")
    @classmethod
    def reject_unsupported_parameters(cls, data):
        if isinstance(data, dict):
            unsupported = sorted(set(data) - set(cls.model_fields))
            if unsupported:
                raise ValueError(
                    f"Unsupported parameter(s): {', '.join(unsupported)}. "
                    f"Supported: {', '.join(cls.model_fields)}."
                )
        return data

    @field_validator("stream")
    @classmethod
    def reject_streaming(cls, v: bool) -> bool:
        if v:
            raise ValueError("stream=true is not supported; omit it or send stream=false")
        return v


@router.post("/v1/chat/completions")
async def create_completion(
    req: CompletionRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(require_api_key),
):
    from src.api.main import get_cache, get_router, get_usage_tracker

    # Sampling parameters the caller left out stay out, so providers can tell
    # "not set" from "set to the default".
    chat_request = ChatCompletionRequest(**req.model_dump(exclude_none=True, exclude={"stream"}))

    cache = get_cache() if settings.cache_enabled else None
    try:
        provider = get_router().route(req.model)

        # Everything that can change the answer goes into the cache key.
        cache_request = {"provider": provider.name, **req.model_dump(mode="json")}
        if cache is not None:
            cached = cache.get(cache_request, tenant=api_key)
            if cached is not None:
                return cached

        response = await provider.complete(chat_request)
    except ProviderTimeoutError as e:
        logger.warning("upstream timeout provider=%s", e.provider)
        raise HTTPException(status_code=504, detail="Upstream provider timed out") from e
    except ProviderError as e:
        # Upstream status and message stay in the logs; callers get a generic 502.
        logger.warning(
            "upstream error provider=%s upstream_status=%s type=%s",
            e.provider,
            e.status_code,
            type(e).__name__,
        )
        raise HTTPException(status_code=502, detail="Upstream provider error") from e
    except Exception as e:
        logger.error("unexpected completion failure type=%s", type(e).__name__)
        raise HTTPException(status_code=500, detail="Internal server error") from e

    response_dict = response.model_dump(by_alias=True)

    if cache is not None:
        cache.set(cache_request, response_dict, tenant=api_key)

    background_tasks.add_task(
        get_usage_tracker().record,
        api_key,
        req.model,
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
    )

    return response_dict
