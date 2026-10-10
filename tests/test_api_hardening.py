"""Request validation, upstream-error mapping and Redis-outage behaviour."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import redis
from fastapi.testclient import TestClient

import src.api.deps as deps_module
import src.api.main as main_module
from src.api.main import app
from src.models.schemas import ChatCompletionResponse, Choice, ChoiceMessage, UsageInfo
from src.providers.base import (
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)

AUTH = {"X-API-Key": "dev-key-1"}
MESSAGES = [{"role": "user", "content": "Hello"}]
SECRET = "sk-upstream-secret-do-not-leak"


def _clear_singletons():
    main_module.get_router.cache_clear()
    main_module.get_cache.cache_clear()
    main_module.get_usage_tracker.cache_clear()
    deps_module.get_rate_limiter.cache_clear()


@pytest.fixture
def redis_mocks():
    """Separate mock Redis clients per component so each can fail on its own."""
    limiter, cache, usage = MagicMock(), MagicMock(), MagicMock()
    limiter.pipeline.return_value.execute.return_value = [None, 0]
    cache.get.return_value = None
    # All three components call the same ``redis.from_url``; build the singletons
    # under one patch, then give each its own client.
    with patch("redis.from_url", return_value=MagicMock()):
        _clear_singletons()
        deps_module.get_rate_limiter()._redis = limiter
        main_module.get_cache()._redis = cache
        main_module.get_usage_tracker()._redis = usage
    try:
        yield {"limiter": limiter, "cache": cache, "usage": usage}
    finally:
        _clear_singletons()


@pytest.fixture
def provider(redis_mocks):
    mock_provider = AsyncMock()
    mock_provider.name = "anthropic"
    mock_provider.complete.return_value = ChatCompletionResponse(
        model="claude-haiku-4-5",
        choices=[Choice(index=0, message=ChoiceMessage(content="ok"))],
        usage=UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )
    gateway = MagicMock()
    gateway.route.return_value = mock_provider
    with patch("src.api.main.get_router", return_value=gateway):
        mock_provider.gateway = gateway
        yield mock_provider


@pytest.fixture
def client():
    return TestClient(app)


def _post(client, **body):
    return client.post("/v1/chat/completions", headers=AUTH, json=body)


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------


class TestRequestValidation:
    def test_empty_messages_is_422_not_500(self, client, provider):
        resp = _post(client, messages=[])
        assert resp.status_code == 422
        provider.complete.assert_not_awaited()

    def test_missing_messages_is_422(self, client, provider):
        assert _post(client, model="claude-haiku-4-5").status_code == 422

    def test_message_with_unknown_role_is_422(self, client, provider):
        resp = _post(client, messages=[{"role": "wizard", "content": "hi"}])
        assert resp.status_code == 422

    def test_message_without_role_is_422(self, client, provider):
        assert _post(client, messages=[{"content": "hi"}]).status_code == 422

    @pytest.mark.parametrize(
        ("field", "value"),
        [("temperature", 3.0), ("temperature", -0.1), ("top_p", 1.5), ("max_tokens", 0)],
    )
    def test_out_of_range_params_are_422(self, client, provider, field, value):
        assert _post(client, messages=MESSAGES, **{field: value}).status_code == 422

    def test_stream_true_is_rejected_not_silently_ignored(self, client, provider):
        resp = _post(client, messages=MESSAGES, stream=True)
        assert resp.status_code == 422
        assert "stream" in resp.text
        provider.complete.assert_not_awaited()

    def test_stream_false_is_accepted(self, client, provider):
        assert _post(client, messages=MESSAGES, stream=False).status_code == 200

    @pytest.mark.parametrize(
        "field",
        ["tools", "tool_choice", "response_format", "n", "seed", "logit_bias", "made_up"],
    )
    def test_unsupported_fields_are_rejected_by_name(self, client, provider, field):
        resp = _post(client, messages=MESSAGES, **{field: 1})
        assert resp.status_code == 422
        assert "Unsupported parameter" in resp.text
        assert field in resp.text
        provider.complete.assert_not_awaited()

    def test_supported_fields_are_forwarded_to_the_provider(self, client, provider):
        resp = _post(
            client, messages=MESSAGES, temperature=0.2, top_p=0.9, stop=["END"], max_tokens=64
        )
        assert resp.status_code == 200
        sent = provider.complete.await_args.args[0]
        assert sent.explicit("temperature") == 0.2
        assert sent.explicit("top_p") == 0.9
        assert sent.stop == ["END"]
        assert sent.max_tokens == 64

    def test_unset_sampling_params_are_not_invented(self, client, provider):
        assert _post(client, messages=MESSAGES).status_code == 200
        sent = provider.complete.await_args.args[0]
        assert sent.explicit("temperature") is None
        assert sent.explicit("top_p") is None
        assert sent.stop is None


# ---------------------------------------------------------------------------
# Upstream error mapping
# ---------------------------------------------------------------------------


class TestUpstreamErrorMapping:
    @pytest.mark.parametrize(
        ("error", "status", "detail"),
        [
            (ProviderTimeoutError(SECRET), 504, "Upstream provider timed out"),
            (ProviderError(SECRET, status_code=400), 502, "Upstream provider error"),
            (ProviderError(SECRET, status_code=500), 502, "Upstream provider error"),
            (ProviderError(SECRET, status_code=529), 502, "Upstream provider error"),
            (ProviderRateLimitError(SECRET), 502, "Upstream provider error"),
            (ProviderAuthError(SECRET), 502, "Upstream provider error"),
        ],
        ids=["timeout", "upstream-400", "upstream-500", "upstream-529", "upstream-429", "auth"],
    )
    def test_provider_errors_map_to_gateway_statuses(self, client, provider, error, status, detail):
        provider.complete.side_effect = error
        resp = _post(client, messages=MESSAGES)
        assert resp.status_code == status
        assert resp.json() == {"detail": detail}
        assert SECRET not in resp.text

    def test_unexpected_exception_is_a_generic_500(self, client, provider):
        provider.complete.side_effect = RuntimeError(SECRET)
        resp = _post(client, messages=MESSAGES)
        assert resp.status_code == 500
        assert resp.json() == {"detail": "Internal server error"}
        assert SECRET not in resp.text

    def test_missing_provider_credentials_is_502_not_500(self, client, provider):
        provider.gateway.route.side_effect = ProviderAuthError("OPENAI_API_KEY is not set")
        resp = _post(client, model="gpt-4o-mini", messages=MESSAGES)
        assert resp.status_code == 502
        assert resp.json() == {"detail": "Upstream provider error"}

    def test_failed_response_is_not_cached(self, client, provider, redis_mocks):
        provider.complete.side_effect = ProviderError("boom")
        _post(client, messages=MESSAGES)
        redis_mocks["cache"].setex.assert_not_called()


# ---------------------------------------------------------------------------
# Redis outage
# ---------------------------------------------------------------------------


class TestRedisOutage:
    def test_rate_limiter_fails_closed_with_503_and_retry_after(
        self, client, provider, redis_mocks
    ):
        redis_mocks["limiter"].pipeline.return_value.execute.side_effect = redis.ConnectionError(
            "Connection refused"
        )
        resp = _post(client, messages=MESSAGES)
        assert resp.status_code == 503
        assert resp.headers["Retry-After"] == "5"
        assert resp.json() == {"detail": "Rate limiter unavailable"}
        provider.complete.assert_not_awaited()

    def test_rate_limiter_record_failure_also_fails_closed(self, client, provider, redis_mocks):
        redis_mocks["limiter"].zadd.side_effect = redis.TimeoutError("slow")
        resp = _post(client, messages=MESSAGES)
        assert resp.status_code == 503
        provider.complete.assert_not_awaited()

    def test_experiments_routes_fail_closed_too(self, client, redis_mocks):
        redis_mocks["limiter"].pipeline.return_value.execute.side_effect = redis.ConnectionError()
        resp = client.get("/v1/experiments", headers=AUTH)
        assert resp.status_code == 503
        assert "Retry-After" in resp.headers

    def test_bad_key_is_still_401_when_redis_is_down(self, client, redis_mocks):
        redis_mocks["limiter"].pipeline.return_value.execute.side_effect = redis.ConnectionError()
        resp = client.post(
            "/v1/chat/completions", headers={"X-API-Key": "nope"}, json={"messages": MESSAGES}
        )
        assert resp.status_code == 401

    def test_cache_outage_degrades_to_a_miss(self, client, provider, redis_mocks):
        redis_mocks["cache"].get.side_effect = redis.ConnectionError("down")
        redis_mocks["cache"].setex.side_effect = redis.ConnectionError("down")
        resp = _post(client, messages=MESSAGES)
        assert resp.status_code == 200
        provider.complete.assert_awaited_once()

    def test_usage_tracker_outage_does_not_fail_the_completion(
        self, client, provider, redis_mocks
    ):
        redis_mocks["usage"].lpush.side_effect = redis.ConnectionError("down")
        assert _post(client, messages=MESSAGES).status_code == 200

    def test_health_does_not_depend_on_redis(self, client, redis_mocks):
        redis_mocks["limiter"].pipeline.return_value.execute.side_effect = redis.ConnectionError()
        assert client.get("/health").status_code == 200


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------


class TestExperimentWeights:
    def _create(self, client, variants):
        return client.post(
            "/v1/experiments", headers=AUTH, json={"id": "weights", "variants": variants}
        )

    def test_weights_summing_to_100_are_accepted(self, client, redis_mocks):
        resp = self._create(
            client, [{"model": "a", "traffic_pct": 70}, {"model": "b", "traffic_pct": 30}]
        )
        assert resp.status_code == 200
        assert resp.json()["variants"] == [
            {"model": "a", "traffic_pct": 70},
            {"model": "b", "traffic_pct": 30},
        ]

    @pytest.mark.parametrize("weights", [(60, 30), (60, 50), (0, 0), (100, 100)])
    def test_weights_not_summing_to_100_are_422(self, client, redis_mocks, weights):
        resp = self._create(
            client, [{"model": f"m{i}", "traffic_pct": w} for i, w in enumerate(weights)]
        )
        assert resp.status_code == 422
        assert "sum to 100" in resp.text

    def test_empty_variants_is_422(self, client, redis_mocks):
        assert self._create(client, []).status_code == 422

    def test_negative_weight_is_422(self, client, redis_mocks):
        resp = self._create(
            client, [{"model": "a", "traffic_pct": 150}, {"model": "b", "traffic_pct": -50}]
        )
        assert resp.status_code == 422

    def test_variant_without_model_is_422(self, client, redis_mocks):
        assert self._create(client, [{"traffic_pct": 100}]).status_code == 422

    def test_rejected_experiment_is_not_stored(self, client, redis_mocks):
        client.post(
            "/v1/experiments",
            headers=AUTH,
            json={"id": "never-stored", "variants": [{"model": "a", "traffic_pct": 5}]},
        )
        listed = client.get("/v1/experiments", headers=AUTH).json()["experiments"]
        assert "never-stored" not in [e["id"] for e in listed]


def test_health_reports_the_package_version(client):
    from src import __version__

    assert __version__ == "1.1.0"
    assert client.get("/health").json() == {"status": "ok", "version": "1.1.0"}
    assert app.version == "1.1.0"
