"""Route-level cache correctness: a cached answer is only ever served for the
same caller asking the same provider/model the same question with the same
response-affecting parameters."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import src.api.deps as deps_module
import src.api.main as main_module
from src.api.main import app
from src.models.schemas import ChatCompletionResponse, Choice, ChoiceMessage, UsageInfo

KEY_A = {"X-API-Key": "dev-key-1"}
KEY_B = {"X-API-Key": "dev-key-2"}
BODY = {"model": "claude-haiku-4-5", "messages": [{"role": "user", "content": "Hello"}]}


class CountingProvider:
    """Stands in for an upstream; counts how often it is actually called."""

    def __init__(self, name: str):
        self.name = name
        self.calls = 0

    async def complete(self, request):
        self.calls += 1
        return ChatCompletionResponse(
            model=request.model,
            choices=[
                Choice(
                    index=0,
                    message=ChoiceMessage(content=f"answer #{self.calls} from {self.name}"),
                )
            ],
            usage=UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )


def _dict_backed_redis() -> MagicMock:
    """Mock Redis whose GET/SETEX really store, so hits and misses are real."""
    store: dict[str, str] = {}
    fake = MagicMock()
    fake.get.side_effect = store.get
    fake.setex.side_effect = lambda key, _ttl, value: store.__setitem__(key, value)
    fake.pipeline.return_value.execute.return_value = [None, 0]
    return fake


def _clear_singletons():
    main_module.get_router.cache_clear()
    main_module.get_cache.cache_clear()
    main_module.get_usage_tracker.cache_clear()
    deps_module.get_rate_limiter.cache_clear()


@pytest.fixture
def upstream():
    providers = {"anthropic": CountingProvider("anthropic"), "openai": CountingProvider("openai")}
    gateway = MagicMock()
    gateway.route.side_effect = lambda model: (
        providers["openai"] if model.startswith("gpt") else providers["anthropic"]
    )
    fake_redis = _dict_backed_redis()
    with (
        patch("src.middleware.rate_limiter.redis.from_url", return_value=fake_redis),
        patch("src.cache.semantic_cache.redis.from_url", return_value=fake_redis),
        patch("src.middleware.usage_tracker.redis.from_url", return_value=fake_redis),
        patch("src.api.main.get_router", return_value=gateway),
    ):
        _clear_singletons()
        try:
            yield providers
        finally:
            _clear_singletons()


@pytest.fixture
def client():
    return TestClient(app)


def _post(client, headers=KEY_A, **overrides):
    resp = client.post("/v1/chat/completions", headers=headers, json={**BODY, **overrides})
    assert resp.status_code == 200, resp.text
    return resp.json()["choices"][0]["message"]["content"]


def test_identical_request_is_a_hit(client, upstream):
    first = _post(client)
    second = _post(client)
    assert first == second == "answer #1 from anthropic"
    assert upstream["anthropic"].calls == 1


def test_different_model_is_a_miss(client, upstream):
    _post(client)
    assert _post(client, model="claude-sonnet-4-6") == "answer #2 from anthropic"
    assert upstream["anthropic"].calls == 2


def test_different_provider_is_a_miss(client, upstream):
    _post(client)
    assert _post(client, model="gpt-4o-mini") == "answer #1 from openai"
    assert upstream["openai"].calls == 1


@pytest.mark.parametrize(
    "param",
    [
        {"max_tokens": 16},
        {"temperature": 0.2},
        {"top_p": 0.5},
        {"stop": ["END"]},
        {"messages": [{"role": "user", "content": "Different question"}]},
        {
            "messages": [
                {"role": "system", "content": "Answer in French"},
                {"role": "user", "content": "Hello"},
            ]
        },
    ],
    ids=["max_tokens", "temperature", "top_p", "stop", "messages", "system-prompt"],
)
def test_different_params_are_a_miss(client, upstream, param):
    _post(client)
    _post(client, **param)
    assert upstream["anthropic"].calls == 2


def test_different_api_key_is_a_miss(client, upstream):
    assert _post(client, headers=KEY_A) == "answer #1 from anthropic"
    assert _post(client, headers=KEY_B) == "answer #2 from anthropic"
    # ...and each caller still gets their own entry back.
    assert _post(client, headers=KEY_A) == "answer #1 from anthropic"
    assert _post(client, headers=KEY_B) == "answer #2 from anthropic"
    assert upstream["anthropic"].calls == 2
