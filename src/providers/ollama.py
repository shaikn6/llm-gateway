"""Ollama local provider — async implementation conforming to LLMProvider."""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator

import httpx

from src.models.schemas import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChoiceDelta,
    ChoiceMessage,
    HealthStatus,
    ModelObject,
    StreamChoice,
    UsageInfo,
)
from src.providers.base import LLMProvider, ProviderError, ProviderTimeoutError

OLLAMA_DEFAULT_URL = "http://localhost:11434"


class OllamaProvider(LLMProvider):
    """Ollama local provider."""

    name = "ollama"

    def __init__(
        self,
        base_url: str = OLLAMA_DEFAULT_URL,
        timeout: float = 120.0,
        api_key: str | None = None,
    ):
        super().__init__(api_key=api_key, timeout=timeout)
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(timeout),
        )

    @staticmethod
    def _build_options(request: ChatCompletionRequest) -> dict:
        """Map the caller's explicit sampling params onto Ollama ``options``."""
        options: dict = {}
        for name in ("temperature", "top_p"):
            if (value := request.explicit(name)) is not None:
                options[name] = value
        if request.stop:
            options["stop"] = [request.stop] if isinstance(request.stop, str) else request.stop
        if request.effective_max_tokens():
            options["num_predict"] = request.effective_max_tokens()
        return options

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        messages = [{"role": m.role.value, "content": m.content or ""} for m in request.messages]
        payload = {
            "model": request.model,
            "messages": messages,
            "stream": False,
        }
        if options := self._build_options(request):
            payload["options"] = options
        try:
            resp = await self._client.post("/api/chat", json=payload)
            resp.raise_for_status()
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(str(exc), provider=self.name) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(str(exc), provider=self.name) from exc

        data = resp.json()
        content = data.get("message", {}).get("content", "")
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            model=request.model,
            choices=[
                Choice(
                    index=0,
                    message=ChoiceMessage(role="assistant", content=content),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(),
        )

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[ChatCompletionChunk]:
        messages = [{"role": m.role.value, "content": m.content or ""} for m in request.messages]
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        payload = {"model": request.model, "messages": messages, "stream": True}
        try:
            async with self._client.stream("POST", "/api/chat", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    import json

                    data = json.loads(line)
                    content = data.get("message", {}).get("content", "")
                    done = data.get("done", False)
                    yield ChatCompletionChunk(
                        id=completion_id,
                        model=request.model,
                        choices=[
                            StreamChoice(
                                index=0,
                                delta=ChoiceDelta(role="assistant", content=content),
                                finish_reason="stop" if done else None,
                            )
                        ],
                    )
                    if done:
                        break
        except httpx.HTTPError as exc:
            raise ProviderError(str(exc), provider=self.name) from exc

    async def list_models(self) -> list[ModelObject]:
        try:
            resp = await self._client.get("/api/tags")
            resp.raise_for_status()
            data = resp.json()
            return [ModelObject(id=m["name"], owned_by="ollama") for m in data.get("models", [])]
        except Exception:  # noqa: BLE001 - listing models is best-effort;
            # any failure (network, malformed response) should degrade to an
            # empty list, not propagate and break a caller that just wants
            # to know what's available.
            return []

    async def health_check(self) -> HealthStatus:
        start = time.monotonic()
        try:
            resp = await self._client.get("/api/tags", timeout=5.0)
            latency_ms = (time.monotonic() - start) * 1000
            healthy = resp.status_code == 200
            return HealthStatus(provider=self.name, healthy=healthy, latency_ms=latency_ms)
        except Exception as exc:  # noqa: BLE001 - a health check must report
            # unhealthy for ANY failure mode, not just the ones we anticipated;
            # narrowing this would risk the health check itself crashing.
            latency_ms = (time.monotonic() - start) * 1000
            return HealthStatus(
                provider=self.name, healthy=False, latency_ms=latency_ms, error=str(exc)
            )
