"""Reject oversized request bodies before they are buffered into memory.

Without this, a single large (or several concurrent) request bodies are read
fully into RAM by the JSON parser, which is a cheap memory-exhaustion DoS for
an authenticated caller. This pure-ASGI middleware enforces a byte cap from
both the declared ``Content-Length`` and the actual streamed bytes (so a
chunked body with no/forged length is also bounded), answering ``413`` before
the route sees the body.
"""

from __future__ import annotations

import json

from starlette.types import ASGIApp, Message, Receive, Scope, Send


class MaxBodySizeMiddleware:
    """Return ``413`` for request bodies larger than ``max_bytes``."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self._app = app
        self._max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        for name, value in scope.get("headers", []):
            if name == b"content-length":
                try:
                    declared = int(value)
                except ValueError:
                    break
                if declared > self._max_bytes:
                    await self._reject(send)
                    return
                break

        seen = 0

        async def guarded_receive() -> Message:
            nonlocal seen
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > self._max_bytes:
                    raise _BodyTooLarge
            return message

        try:
            await self._app(scope, guarded_receive, send)
        except _BodyTooLarge:
            await self._reject(send)

    async def _reject(self, send: Send) -> None:
        body = json.dumps(
            {"detail": f"Request body exceeds the {self._max_bytes}-byte limit"}
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


class _BodyTooLarge(Exception):
    """Internal signal that the streamed body passed the cap."""
