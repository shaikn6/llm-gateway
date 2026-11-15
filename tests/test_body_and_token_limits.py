"""Request-body size cap and max_tokens upper-bound hardening."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from src.api.routes.completions import CompletionRequest
from src.middleware.body_limit import MaxBodySizeMiddleware
from src.models.schemas import MAX_OUTPUT_TOKENS

MESSAGES = [{"role": "user", "content": "Hello"}]


def _echo_app(max_bytes: int) -> TestClient:
    async def echo(request):
        await request.body()
        return JSONResponse({"ok": True})

    inner = Starlette(routes=[Route("/", echo, methods=["POST"])])
    inner.add_middleware(MaxBodySizeMiddleware, max_bytes=max_bytes)
    return TestClient(inner)


def test_body_within_limit_passes():
    client = _echo_app(max_bytes=100)
    assert client.post("/", content=b"x" * 50).status_code == 200


def test_body_over_declared_length_rejected():
    client = _echo_app(max_bytes=100)
    resp = client.post("/", content=b"x" * 500)
    assert resp.status_code == 413
    assert "limit" in resp.json()["detail"]


def test_body_over_streamed_bytes_rejected():
    # A chunked body sends no Content-Length, so the cap must be enforced on
    # the actual streamed bytes, not just the declared length.
    client = _echo_app(max_bytes=100)

    def chunks():
        for _ in range(10):
            yield b"x" * 50

    resp = client.post("/", content=chunks())
    assert resp.status_code == 413


def test_max_tokens_above_ceiling_rejected():
    # Validation lives on the request model, so an absurd value is a 422 at
    # the route rather than being forwarded upstream.
    with pytest.raises(ValidationError):
        CompletionRequest(max_tokens=999999999999, messages=MESSAGES)


def test_max_tokens_at_ceiling_accepted():
    req = CompletionRequest(max_tokens=MAX_OUTPUT_TOKENS, messages=MESSAGES)
    assert req.max_tokens == MAX_OUTPUT_TOKENS


async def _run_with_content_length(cl_value: bytes, body: bytes):
    # Drive the ASGI middleware directly: httpx recomputes Content-Length, so a
    # malformed header value can only be injected at the scope level.
    sent = []

    async def inner(scope, receive, send):
        await receive()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    mw = MaxBodySizeMiddleware(inner, max_bytes=100)
    scope = {"type": "http", "headers": [(b"content-length", cl_value)]}

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message):
        sent.append(message)

    await mw(scope, receive, send)
    return sent


async def test_non_integer_content_length_small_body_passes():
    # A malformed Content-Length must not crash the cap: it is ignored and the
    # request proceeds, enforced on the actual streamed bytes instead.
    sent = await _run_with_content_length(b"not-a-number", b"x" * 10)
    assert sent[0]["status"] == 200


async def test_non_integer_content_length_oversize_still_rejected():
    # Ignoring the bad header must not disable the cap: an oversize streamed
    # body is still rejected with 413.
    sent = await _run_with_content_length(b"not-a-number", b"x" * 500)
    assert sent[0]["status"] == 413
