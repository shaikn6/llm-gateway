"""POST /v1/chat/completions — OpenAI-compatible endpoint."""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator, model_validator

from src.api.deps import require_api_key
from src.api.routes.experiments import tenant_ab_router
from src.config import settings
from src.middleware.metrics import metrics
from src.middleware.usage_tracker import request_cost_usd
from src.models.schemas import MAX_OUTPUT_TOKENS, ChatCompletionRequest, ChatMessage
from src.providers.base import ProviderError, ProviderTimeoutError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["completions"])


class CompletionRequest(BaseModel):
    """The subset of the OpenAI chat-completions request this gateway honours.

    Anything else is rejected rather than accepted and silently dropped.
    """

    model: str = "claude-haiku-4-5"
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int = Field(default=1024, ge=1, le=MAX_OUTPUT_TOKENS)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    stop: str | list[str] | None = None
    stream: bool = False
    # Opt-in A/B routing: when set, the experiment's assignment for ``user``
    # (same bucketing as GET /v1/experiments/{id}/assignment) overrides ``model``.
    experiment: str | None = None
    user: str | None = None

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

    model = req.model
    if req.experiment is not None:
        try:
            model = tenant_ab_router(api_key).get_assignment(
                req.experiment, req.user or "default"
            )
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

    # Sampling parameters the caller left out stay out, so providers can tell
    # "not set" from "set to the default".
    chat_request = ChatCompletionRequest(
        **{
            **req.model_dump(exclude_none=True, exclude={"stream", "experiment"}),
            "model": model,
        }
    )

    cache = get_cache() if settings.cache_enabled else None
    started = time.perf_counter()
    try:
        provider = get_router().route(model)

        # Everything that can change the answer goes into the cache key.
        cache_request = {"provider": provider.name, **req.model_dump(mode="json"), "model": model}
        if cache is not None:
            cached = cache.get(cache_request, tenant=api_key)
            if cached is not None:
                metrics.inc("llm_gateway_cache_total", result="hit")
                # Nothing was sent upstream, so nothing was billed for this call.
                return {
                    **cached,
                    "x-gateway-cache": "hit",
                    "x-gateway-cost-usd": 0.0,
                    "x-gateway-latency-ms": round((time.perf_counter() - started) * 1000),
                }
            metrics.inc("llm_gateway_cache_total", result="miss")

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

    prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens
    cost_usd = request_cost_usd(model, prompt_tokens, completion_tokens)

    # Same per-model counts the UsageTracker records below, also exposed on /metrics.
    metrics.inc("llm_gateway_model_tokens_total", prompt_tokens, model=model, type="prompt")
    metrics.inc("llm_gateway_model_tokens_total", completion_tokens, model=model, type="completion")
    if cost_usd is not None:
        metrics.inc("llm_gateway_model_cost_usd_total", cost_usd, model=model)

    response_dict = response.model_dump(by_alias=True)
    response_dict["x-gateway-provider"] = provider.name
    response_dict["x-gateway-cache"] = "miss" if cache is not None else "disabled"
    response_dict["x-gateway-cost-usd"] = cost_usd
    response_dict["x-gateway-latency-ms"] = round((time.perf_counter() - started) * 1000)
    if req.experiment is not None:
        response_dict["x-gateway-experiment"] = req.experiment
        response_dict["x-gateway-variant"] = model

    if cache is not None:
        cache.set(cache_request, response_dict, tenant=api_key)

    background_tasks.add_task(
        get_usage_tracker().record,
        api_key,
        model,
        prompt_tokens,
        completion_tokens,
    )

    return response_dict
