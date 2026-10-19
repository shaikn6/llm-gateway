"""Request validation, upstream-error mapping and Redis-outage behaviour."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import redis
from fastapi.testclient import TestClient

import src.api.deps as deps_module
import src.api.main as main_module
from src.api.main import app
from src.middleware.audit import key_fingerprint
from src.models.schemas import ChatCompletionResponse, Choice, ChoiceMessage, UsageInfo
from src.providers.base import (
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)

AUTH = {"X-API-Key": "dev-key-1"}
AUTH_B = {"X-API-Key": "dev-key-2"}
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
    limiter.eval.return_value = [1, 99]  # allowed
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
        redis_mocks["limiter"].eval.side_effect = redis.ConnectionError("Connection refused")
        resp = _post(client, messages=MESSAGES)
        assert resp.status_code == 503
        assert resp.headers["Retry-After"] == "5"
        assert resp.json() == {"detail": "Rate limiter unavailable"}
        provider.complete.assert_not_awaited()

    def test_rate_limiter_timeout_also_fails_closed(self, client, provider, redis_mocks):
        redis_mocks["limiter"].eval.side_effect = redis.TimeoutError("slow")
        resp = _post(client, messages=MESSAGES)
        assert resp.status_code == 503
        provider.complete.assert_not_awaited()

    def test_experiments_routes_fail_closed_too(self, client, redis_mocks):
        redis_mocks["limiter"].eval.side_effect = redis.ConnectionError()
        resp = client.get("/v1/experiments", headers=AUTH)
        assert resp.status_code == 503
        assert "Retry-After" in resp.headers

    def test_bad_key_is_still_401_when_redis_is_down(self, client, redis_mocks):
        redis_mocks["limiter"].eval.side_effect = redis.ConnectionError()
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
        redis_mocks["limiter"].eval.side_effect = redis.ConnectionError()
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


# ---------------------------------------------------------------------------
# A/B routing on /v1/chat/completions
# ---------------------------------------------------------------------------


class TestExperimentRouting:
    def _experiment(self, client, exp_id, variants):
        resp = client.post("/v1/experiments", headers=AUTH, json={"id": exp_id, "variants": variants})
        assert resp.status_code == 200

    def test_request_is_served_by_the_assigned_variant_and_records_it(
        self, client, provider, redis_mocks
    ):
        self._experiment(client, "route-all", [{"model": "gpt-4o-mini", "traffic_pct": 100}])
        resp = _post(
            client, model="claude-haiku-4-5", messages=MESSAGES, experiment="route-all", user="u1"
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["x-gateway-experiment"] == "route-all"
        assert body["x-gateway-variant"] == "gpt-4o-mini"
        provider.gateway.route.assert_called_once_with("gpt-4o-mini")
        sent = provider.complete.await_args.args[0]
        assert sent.model == "gpt-4o-mini"
        assert not hasattr(sent, "experiment")
        recorded = json.loads(redis_mocks["usage"].lpush.call_args.args[1])
        assert recorded["model"] == "gpt-4o-mini"

    def test_assignment_matches_the_assignment_endpoint(self, client, provider, redis_mocks):
        self._experiment(
            client,
            "route-split",
            [{"model": "gpt-4o-mini", "traffic_pct": 50}, {"model": "claude-haiku-4-5", "traffic_pct": 50}],
        )
        served = set()
        for user in (f"user{i}" for i in range(20)):
            expected = client.get(
                f"/v1/experiments/route-split/assignment?user_id={user}", headers=AUTH
            ).json()["model"]
            resp = _post(client, messages=MESSAGES, experiment="route-split", user=user)
            assert resp.json()["x-gateway-variant"] == expected
            served.add(expected)
        assert served == {"gpt-4o-mini", "claude-haiku-4-5"}

    def test_unknown_experiment_is_404_and_calls_no_provider(self, client, provider, redis_mocks):
        resp = _post(client, messages=MESSAGES, experiment="no-such-experiment")
        assert resp.status_code == 404
        provider.complete.assert_not_awaited()

    def test_requests_without_an_experiment_are_unchanged(self, client, provider, redis_mocks):
        resp = _post(client, model="gpt-4o", messages=MESSAGES)
        assert resp.status_code == 200
        assert "x-gateway-experiment" not in resp.json()
        provider.gateway.route.assert_called_once_with("gpt-4o")
        assert json.loads(redis_mocks["usage"].lpush.call_args.args[1])["model"] == "gpt-4o"

    def test_cache_key_uses_the_assigned_model(self, client, provider, redis_mocks):
        self._experiment(client, "route-cache", [{"model": "gpt-4o-mini", "traffic_pct": 100}])
        with patch.object(main_module.get_cache(), "get", return_value=None) as cache_get:
            _post(client, model="claude-haiku-4-5", messages=MESSAGES, experiment="route-cache")
        assert cache_get.call_args.args[0]["model"] == "gpt-4o-mini"


class TestExperimentTenantIsolation:
    """Experiments belong to the API key that created them."""

    def _create(self, client, headers, exp_id, model):
        resp = client.post(
            "/v1/experiments",
            headers=headers,
            json={"id": exp_id, "variants": [{"model": model, "traffic_pct": 100}]},
        )
        assert resp.status_code == 200

    def test_other_tenant_gets_404_on_assignment_and_list(self, client, redis_mocks):
        self._create(client, AUTH, "iso-a", "gpt-4o-mini")
        url = "/v1/experiments/iso-a/assignment"
        assert client.get(url, headers=AUTH).status_code == 200
        assert client.get(url, headers=AUTH_B).status_code == 404
        listed = client.get("/v1/experiments", headers=AUTH_B).json()["experiments"]
        assert "iso-a" not in [e["id"] for e in listed]

    def test_other_tenant_cannot_route_completions_by_it(self, client, provider, redis_mocks):
        self._create(client, AUTH, "iso-route", "gpt-4o-mini")
        resp = client.post(
            "/v1/chat/completions",
            headers=AUTH_B,
            json={"messages": MESSAGES, "experiment": "iso-route"},
        )
        assert resp.status_code == 404
        provider.complete.assert_not_awaited()
        redis_mocks["usage"].lpush.assert_not_called()
        # The owner is unaffected.
        assert _post(client, messages=MESSAGES, experiment="iso-route").status_code == 200

    def test_same_id_for_another_tenant_does_not_overwrite(self, client, redis_mocks):
        self._create(client, AUTH, "iso-same", "gpt-4o-mini")
        self._create(client, AUTH_B, "iso-same", "claude-haiku-4-5")
        url = "/v1/experiments/iso-same/assignment"
        assert client.get(url, headers=AUTH).json()["model"] == "gpt-4o-mini"
        assert client.get(url, headers=AUTH_B).json()["model"] == "claude-haiku-4-5"

    def test_same_tenant_recreate_still_overwrites(self, client, redis_mocks):
        self._create(client, AUTH, "iso-redo", "gpt-4o-mini")
        self._create(client, AUTH, "iso-redo", "claude-haiku-4-5")
        url = "/v1/experiments/iso-redo/assignment"
        assert client.get(url, headers=AUTH).json()["model"] == "claude-haiku-4-5"


# ---------------------------------------------------------------------------
# GET /v1/usage
# ---------------------------------------------------------------------------


class TestUsageEndpoint:
    def test_returns_recorded_usage_for_the_calling_key(self, client, redis_mocks):
        redis_mocks["usage"].lrange.return_value = [
            json.dumps({"model": "gpt-4o-mini", "prompt_tokens": 1000, "completion_tokens": 500}),
            json.dumps({"model": "gpt-4o-mini", "prompt_tokens": 1000, "completion_tokens": 500}),
        ]
        resp = client.get("/v1/usage", headers=AUTH)
        assert resp.status_code == 200
        assert resp.json() == {
            "total_requests": 2,
            "input_tokens": 2000,
            "output_tokens": 1000,
            "cost_usd": 0.0009,
            "by_model": {"gpt-4o-mini": pytest.approx(0.0009)},
        }
        redis_mocks["usage"].lrange.assert_called_once_with(
            f"usage:{key_fingerprint('dev-key-1')}", 0, -1
        )

    def test_is_scoped_to_the_calling_key(self, client, redis_mocks):
        client.get("/v1/usage", headers={"X-API-Key": "dev-key-2"})
        redis_mocks["usage"].lrange.assert_called_once_with(
            f"usage:{key_fingerprint('dev-key-2')}", 0, -1
        )

    def test_empty_usage_is_zeros(self, client, redis_mocks):
        redis_mocks["usage"].lrange.return_value = []
        assert client.get("/v1/usage", headers=AUTH).json()["total_requests"] == 0

    @pytest.mark.parametrize("headers", [{}, {"X-API-Key": "nope"}])
    def test_requires_a_valid_api_key(self, client, redis_mocks, headers):
        assert client.get("/v1/usage", headers=headers).status_code == 401
        redis_mocks["usage"].lrange.assert_not_called()

    def test_usage_store_outage_is_503(self, client, redis_mocks):
        redis_mocks["usage"].lrange.side_effect = redis.ConnectionError("down")
        resp = client.get("/v1/usage", headers=AUTH)
        assert resp.status_code == 503
        assert resp.json() == {"detail": "Usage store unavailable"}
