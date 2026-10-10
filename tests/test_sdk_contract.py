"""Contract tests: the kwargs each provider builds must be accepted by the
REAL installed SDK method, not by a permissive mock.

``AsyncMock`` accepts any keyword argument, so a provider can pass a parameter
the SDK has removed (as happened with ``temperature``/``top_p`` on
``anthropic`` 1.x) and every mocked test still passes. These tests bind the
exact kwargs against the real method signatures and then drive the real SDK
request path against an in-process fake transport. No network, no API keys.
"""

from __future__ import annotations

import inspect
import json

import httpx2
import pytest

from src.models.schemas import ChatCompletionRequest
from src.providers.anthropic import AnthropicProvider
from src.providers.openai import OpenAIAsyncProvider

REQUEST_VARIANTS = [
    pytest.param({}, id="defaults-only"),
    pytest.param({"temperature": 0.2}, id="temperature"),
    pytest.param({"top_p": 0.9}, id="top_p"),
    pytest.param({"stop": "END"}, id="stop-str"),
    pytest.param({"stop": ["A", "B"]}, id="stop-list"),
    pytest.param({"max_tokens": 64}, id="max_tokens"),
    pytest.param(
        {"temperature": 0.2, "top_p": 0.9, "stop": ["A"], "max_tokens": 64},
        id="everything",
    ),
]


def _request(model: str, **overrides) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model=model,
        messages=[
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "Hello"},
        ],
        **overrides,
    )


def _bind(method, kwargs: dict) -> None:
    """Raise TypeError exactly as calling ``method(**kwargs)`` would."""
    inspect.signature(method).bind(**kwargs)


# ---------------------------------------------------------------------------
# Signature binding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("overrides", REQUEST_VARIANTS)
def test_anthropic_kwargs_bind_to_real_create_and_stream(overrides):
    provider = AnthropicProvider(api_key="sk-ant-not-a-real-key")
    kwargs = provider._build_kwargs(_request("claude-haiku-4-5", **overrides))
    _bind(provider._client.messages.create, kwargs)
    _bind(provider._client.messages.stream, kwargs)


@pytest.mark.parametrize("overrides", REQUEST_VARIANTS)
def test_openai_kwargs_bind_to_real_create(overrides):
    provider = OpenAIAsyncProvider(api_key="sk-not-a-real-key")
    kwargs = provider._build_kwargs(_request("gpt-4o-mini", **overrides))
    _bind(provider._client.chat.completions.create, kwargs)
    _bind(provider._client.chat.completions.create, {**kwargs, "stream": True})


def test_anthropic_omits_sampling_params_the_caller_did_not_set():
    provider = AnthropicProvider(api_key="sk-ant-not-a-real-key")
    kwargs = provider._build_kwargs(_request("claude-haiku-4-5"))
    assert "temperature" not in kwargs
    assert "top_p" not in kwargs
    assert "extra_body" not in kwargs


def test_openai_omits_sampling_params_the_caller_did_not_set():
    provider = OpenAIAsyncProvider(api_key="sk-not-a-real-key")
    kwargs = provider._build_kwargs(_request("gpt-4o-mini"))
    assert set(kwargs) == {"model", "messages"}


# ---------------------------------------------------------------------------
# Real SDK request path against a fake transport
# ---------------------------------------------------------------------------


def _capturing_transport(response_json: dict, seen: list[dict]) -> httpx2.MockTransport:
    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(json.loads(request.content))
        return httpx2.Response(200, json=response_json)

    return httpx2.MockTransport(handler)


ANTHROPIC_REPLY = {
    "id": "msg_fake",
    "type": "message",
    "role": "assistant",
    "model": "claude-haiku-4-5",
    "content": [{"type": "text", "text": "hi"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 3, "output_tokens": 2},
}

OPENAI_REPLY = {
    "id": "chatcmpl-fake",
    "object": "chat.completion",
    "created": 0,
    "model": "gpt-4o-mini",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "hi"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
}


@pytest.mark.asyncio
async def test_anthropic_complete_runs_through_the_real_sdk():
    import anthropic

    seen: list[dict] = []
    provider = AnthropicProvider(api_key="sk-ant-not-a-real-key")
    provider._client = anthropic.AsyncAnthropic(
        api_key="sk-ant-not-a-real-key",
        http_client=httpx2.AsyncClient(transport=_capturing_transport(ANTHROPIC_REPLY, seen)),
    )

    resp = await provider.complete(_request("claude-haiku-4-5"))

    assert resp.choices[0].message.content == "hi"
    assert resp.usage.total_tokens == 5
    assert "temperature" not in seen[0]
    assert "top_p" not in seen[0]


@pytest.mark.asyncio
async def test_anthropic_explicit_sampling_params_reach_the_wire():
    import anthropic

    seen: list[dict] = []
    provider = AnthropicProvider(api_key="sk-ant-not-a-real-key")
    provider._client = anthropic.AsyncAnthropic(
        api_key="sk-ant-not-a-real-key",
        http_client=httpx2.AsyncClient(transport=_capturing_transport(ANTHROPIC_REPLY, seen)),
    )

    await provider.complete(_request("claude-haiku-4-5", temperature=0.2, top_p=0.9, stop="END"))

    assert seen[0]["temperature"] == 0.2
    assert seen[0]["top_p"] == 0.9
    assert seen[0]["stop_sequences"] == ["END"]
    assert seen[0]["system"] == "be brief"


@pytest.mark.asyncio
async def test_openai_complete_runs_through_the_real_sdk():
    import openai

    seen: list[dict] = []
    provider = OpenAIAsyncProvider(api_key="sk-not-a-real-key")
    provider._client = openai.AsyncOpenAI(
        api_key="sk-not-a-real-key",
        http_client=httpx2.AsyncClient(transport=_capturing_transport(OPENAI_REPLY, seen)),
    )

    resp = await provider.complete(
        _request("gpt-4o-mini", temperature=0.2, top_p=0.9, stop=["A"], max_tokens=64)
    )

    assert resp.choices[0].message.content == "hi"
    assert seen[0]["temperature"] == 0.2
    assert seen[0]["top_p"] == 0.9
    assert seen[0]["stop"] == ["A"]
    assert seen[0]["max_tokens"] == 64
