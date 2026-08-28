"""Structured audit logging.

Every request that reaches the gateway is logged as a single JSON line with the
fields an operator needs to reconstruct "who called what, when, and how it went"
— without ever recording prompt or completion content, which for an LLM gateway
is the sensitive part.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.types import ASGIApp

from src.config import settings

_AUDIT_LOGGER = "llm_gateway.audit"


def key_fingerprint(api_key: str) -> str:
    """Short, non-reversible tag for an API key — safe to write to logs."""
    return hashlib.sha256(api_key.encode()).hexdigest()[:12]


class JsonFormatter(logging.Formatter):
    """Render a log record (plus its ``audit`` extra) as one JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        payload.update(getattr(record, "audit", {}))
        return json.dumps(payload, separators=(",", ":"))


def configure_logging() -> None:
    """Point the root logger at a single JSON handler on stdout."""
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(settings.log_level.upper())


class AuditMiddleware(BaseHTTPMiddleware):
    """Emit one structured audit record per request."""

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)
        self._log = logging.getLogger(_AUDIT_LOGGER)

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        api_key = request.headers.get("x-api-key", "")
        started = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            self._log.info(
                "request",
                extra={
                    "audit": {
                        "request_id": request_id,
                        "method": request.method,
                        "path": request.url.path,
                        "status": status_code,
                        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                        "client": request.client.host if request.client else None,
                        "api_key": key_fingerprint(api_key) if api_key else None,
                    }
                },
            )
