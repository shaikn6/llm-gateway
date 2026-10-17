"""GET /v1/usage — the calling key's recorded token usage and cost."""

import logging

import redis
from fastapi import APIRouter, Depends, HTTPException

from src.api.deps import require_api_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/usage", tags=["usage"])


@router.get("")
def get_usage(api_key: str = Depends(require_api_key)):
    from src.api.main import get_usage_tracker

    # Scoped to the caller: usage is keyed by the authenticated key, so one
    # tenant can never read another's totals.
    try:
        return get_usage_tracker().get_usage(api_key)
    except redis.RedisError as e:
        logger.warning("usage read failed (%s)", type(e).__name__)
        raise HTTPException(status_code=503, detail="Usage store unavailable") from e
