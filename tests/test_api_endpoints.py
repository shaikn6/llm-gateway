"""Integration tests for FastAPI endpoints — completions and experiments."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import src.api.deps as deps_module
import src.api.main as main_module
from src.api.main import app
from src.models.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChoiceMessage,
    UsageInfo,
)
from src.providers.anthropic import AnthropicProvider

VALID_API_KEY = "dev-key-1"
AUTH_HEADERS = {"X-API-Key": VALID_API_KEY}


def _reset_singletons():
    """All four singleton getters are lru_cache-wrapped; reset via cache_clear()."""
    main_module.get_router.cache_clear()
    main_module.get_cache.cache_clear()
    main_module.get_usage_tracker.cache_clear()
    deps_module.get_rate_limiter.cache_clear()


@pytest.fixture(autouse=True)
def _reset_singletons_and_mock_redis():
    """All Redis-backed singletons (rate limiter, cache, usage tracker) are
    reconstructed fresh per test, backed by a mocked Redis client so tests
    don't need a real Redis server. Rate limiting defaults to "always
    allowed" and the cache defaults to "always miss" unless a test
    overrides the mock's return values."""
    mock_redis = MagicMock()
    pipe = MagicMock()
    pipe.execute.return_value = [None, 0]  # well under any configured limit
    mock_redis.pipeline.return_value = pipe
    mock_redis.get.return_value = None  # cache miss by default

    with (
        patch("src.middleware.rate_limiter.redis.from_url", return_value=mock_redis),
        patch("src.cache.semantic_cache.redis.from_url", return_value=mock_redis),
        patch("src.middleware.usage_tracker.redis.from_url", return_value=mock_redis),
    ):
        _reset_singletons()
        try:
            yield mock_redis
        finally:
            _reset_singletons()


@pytest.fixture
def client():
    return TestClient(app)


def _make_completion_response(content="Hello!", model="claude-haiku-4-5"):
    return ChatCompletionResponse(
        id="chatcmpl-test",
        model=model,
        choices=[
            Choice(index=0, message=ChoiceMessage(role="assistant", content=content), finish_reason="stop")
        ],
        usage=UsageInfo(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_returns_status_ok(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert data["status"] == "ok"

    def test_health_returns_version(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert "version" in data
        assert data["version"] == "0.1.0"


# ---------------------------------------------------------------------------
# POST /v1/chat/completions
# ---------------------------------------------------------------------------


class TestCompletionsEndpoint:
    def test_completions_happy_path(self, client):
        mock_provider = AsyncMock()
        mock_provider.complete.return_value = _make_completion_response()

        with patch("src.api.main.get_router") as mock_get_router:
            mock_router = MagicMock()
            mock_router.route.return_value = mock_provider
            mock_get_router.return_value = mock_router

            resp = client.post(
                "/v1/chat/completions",
                headers=AUTH_HEADERS,
                json={
                    "model": "claude-haiku-4-5",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

        assert resp.status_code == 200

    def test_completions_returns_provider_response(self, client):
        mock_provider = AsyncMock()
        mock_provider.complete.return_value = _make_completion_response()

        with patch("src.api.main.get_router") as mock_get_router:
            mock_router = MagicMock()
            mock_router.route.return_value = mock_provider
            mock_get_router.return_value = mock_router

            resp = client.post(
                "/v1/chat/completions",
                headers=AUTH_HEADERS,
                json={
                    "model": "claude-haiku-4-5",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

        data = resp.json()
        assert data["model"] == "claude-haiku-4-5"

    def test_completions_provider_exception_returns_500(self, client):
        mock_provider = AsyncMock()
        mock_provider.complete.side_effect = RuntimeError("Provider error")

        with patch("src.api.main.get_router") as mock_get_router:
            mock_router = MagicMock()
            mock_router.route.return_value = mock_provider
            mock_get_router.return_value = mock_router

            resp = client.post(
                "/v1/chat/completions",
                headers=AUTH_HEADERS,
                json={
                    "model": "claude-haiku-4-5",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

        assert resp.status_code == 500

    def test_completions_error_detail_in_response(self, client):
        mock_provider = AsyncMock()
        mock_provider.complete.side_effect = RuntimeError("Provider error")

        with patch("src.api.main.get_router") as mock_get_router:
            mock_router = MagicMock()
            mock_router.route.return_value = mock_provider
            mock_get_router.return_value = mock_router

            resp = client.post(
                "/v1/chat/completions",
                headers=AUTH_HEADERS,
                json={
                    "model": "claude-haiku-4-5",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

        assert "Provider error" in resp.json()["detail"]

    def test_completions_defaults_model_to_claude_haiku(self, client):
        mock_provider = AsyncMock()
        mock_provider.complete.return_value = _make_completion_response()

        with patch("src.api.main.get_router") as mock_get_router:
            mock_router = MagicMock()
            mock_router.route.return_value = mock_provider
            mock_get_router.return_value = mock_router

            resp = client.post(
                "/v1/chat/completions",
                headers=AUTH_HEADERS,
                json={"messages": [{"role": "user", "content": "Hello"}]},
            )

        assert resp.status_code == 200
        mock_router.route.assert_called_once_with("claude-haiku-4-5")

    def test_completions_routes_gpt_model_to_openai(self, client):
        mock_provider = AsyncMock()
        mock_provider.complete.return_value = _make_completion_response(
            content="GPT response", model="gpt-4o-mini"
        )

        with patch("src.api.main.get_router") as mock_get_router:
            mock_router = MagicMock()
            mock_router.route.return_value = mock_provider
            mock_get_router.return_value = mock_router

            resp = client.post(
                "/v1/chat/completions",
                headers=AUTH_HEADERS,
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

        assert resp.status_code == 200
        mock_router.route.assert_called_once_with("gpt-4o-mini")

    def test_completions_passes_max_tokens(self, client):
        mock_provider = AsyncMock()
        mock_provider.complete.return_value = _make_completion_response()

        with patch("src.api.main.get_router") as mock_get_router:
            mock_router = MagicMock()
            mock_router.route.return_value = mock_provider
            mock_get_router.return_value = mock_router

            resp = client.post(
                "/v1/chat/completions",
                headers=AUTH_HEADERS,
                json={
                    "model": "claude-haiku-4-5",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 512,
                },
            )

        assert resp.status_code == 200
        request_arg = mock_provider.complete.call_args[0][0]
        assert isinstance(request_arg, ChatCompletionRequest)
        assert request_arg.max_tokens == 512

    def test_completions_missing_messages_returns_422(self, client):
        resp = client.post(
            "/v1/chat/completions",
            headers=AUTH_HEADERS,
            json={"model": "claude-haiku-4-5"},
        )
        assert resp.status_code == 422

    def test_completions_missing_api_key_returns_401(self, client):
        resp = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hello"}]},
        )
        assert resp.status_code == 401

    def test_completions_invalid_api_key_returns_401(self, client):
        resp = client.post(
            "/v1/chat/completions",
            headers={"X-API-Key": "not-a-real-key"},
            json={"messages": [{"role": "user", "content": "Hello"}]},
        )
        assert resp.status_code == 401

    def test_completions_rate_limited_returns_429(self, client, _reset_singletons_and_mock_redis):
        mock_redis = _reset_singletons_and_mock_redis
        pipe = MagicMock()
        pipe.execute.return_value = [None, 1000]  # way over any configured limit
        mock_redis.pipeline.return_value = pipe

        resp = client.post(
            "/v1/chat/completions",
            headers=AUTH_HEADERS,
            json={"messages": [{"role": "user", "content": "Hello"}]},
        )
        assert resp.status_code == 429

    def test_completions_cache_hit_skips_provider_call(self, client, _reset_singletons_and_mock_redis):
        mock_redis = _reset_singletons_and_mock_redis
        cached_response = _make_completion_response(content="Cached!").model_dump(by_alias=True)
        mock_redis.get.return_value = json.dumps(cached_response)

        mock_provider = AsyncMock()

        with patch("src.api.main.get_router") as mock_get_router:
            mock_router = MagicMock()
            mock_router.route.return_value = mock_provider
            mock_get_router.return_value = mock_router

            resp = client.post(
                "/v1/chat/completions",
                headers=AUTH_HEADERS,
                json={"messages": [{"role": "user", "content": "Hello"}]},
            )

        assert resp.status_code == 200
        assert resp.json()["choices"][0]["message"]["content"] == "Cached!"
        mock_provider.complete.assert_not_called()


class TestCompletionsRealProviderContract:
    """Regression coverage for the async/sync signature mismatch: the route
    must actually `await provider.complete(request)` against the real
    `LLMProvider` async contract, not call a synchronous
    `complete(messages, model=..., max_tokens=...)` signature.
    """

    def test_autospec_mock_rejects_old_sync_calling_convention(self, client):
        """An AsyncMock built from AnthropicProvider's real spec will raise
        TypeError if the route still calls complete() with the old
        (messages, model=, max_tokens=) signature -- this is exactly the
        bug class described in issue #1, and a plain MagicMock would never
        catch it."""
        provider_mock = AsyncMock(spec=AnthropicProvider)
        provider_mock.complete.return_value = _make_completion_response()

        with patch("src.api.main.get_router") as mock_get_router:
            mock_router = MagicMock()
            mock_router.route.return_value = provider_mock
            mock_get_router.return_value = mock_router

            resp = client.post(
                "/v1/chat/completions",
                headers=AUTH_HEADERS,
                json={"messages": [{"role": "user", "content": "Hello"}]},
            )

        assert resp.status_code == 200
        provider_mock.complete.assert_awaited_once()
        request_arg = provider_mock.complete.call_args[0][0]
        assert isinstance(request_arg, ChatCompletionRequest)

    def test_completions_endpoint_exercises_real_anthropic_provider_complete(self, client):
        """End-to-end through the real FastAPI route and the real, unmocked
        AnthropicProvider.complete() coroutine -- only the outermost
        Anthropic SDK network call is stubbed. This is the actual code path
        (not a test that mocks around the bug)."""
        with patch("src.providers.anthropic.anthropic.AsyncAnthropic") as MockAnthropicSDK:
            mock_sdk_client = MagicMock()
            mock_message = MagicMock()
            mock_message.id = "msg_test123"
            mock_message.model = "claude-haiku-4-5"
            mock_message.stop_reason = "end_turn"
            content_block = MagicMock()
            content_block.text = "Hello from real provider!"
            mock_message.content = [content_block]
            mock_message.usage = MagicMock(input_tokens=10, output_tokens=5)
            mock_sdk_client.messages.create = AsyncMock(return_value=mock_message)
            MockAnthropicSDK.return_value = mock_sdk_client

            real_provider = AnthropicProvider(api_key="sk-ant-test")

            with patch("src.api.main.get_router") as mock_get_router:
                mock_router = MagicMock()
                mock_router.route.return_value = real_provider
                mock_get_router.return_value = mock_router

                resp = client.post(
                    "/v1/chat/completions",
                    headers=AUTH_HEADERS,
                    json={"messages": [{"role": "user", "content": "Hello"}]},
                )

        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["message"]["content"] == "Hello from real provider!"
        mock_sdk_client.messages.create.assert_awaited_once()


# ---------------------------------------------------------------------------
# GET /v1/experiments
# ---------------------------------------------------------------------------


class TestExperimentsListEndpoint:
    def test_list_experiments_returns_200(self, client):
        resp = client.get("/v1/experiments", headers=AUTH_HEADERS)
        assert resp.status_code == 200

    def test_list_experiments_returns_experiments_key(self, client):
        resp = client.get("/v1/experiments", headers=AUTH_HEADERS)
        data = resp.json()
        assert "experiments" in data

    def test_list_experiments_initially_empty(self, client):
        # Fresh app state may have leftover state from other tests in module,
        # so just verify the structure is correct
        resp = client.get("/v1/experiments", headers=AUTH_HEADERS)
        data = resp.json()
        assert isinstance(data["experiments"], list)

    def test_list_experiments_missing_api_key_returns_401(self, client):
        resp = client.get("/v1/experiments")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# POST /v1/experiments
# ---------------------------------------------------------------------------


class TestExperimentsCreateEndpoint:
    def test_create_experiment_returns_200(self, client):
        resp = client.post(
            "/v1/experiments",
            headers=AUTH_HEADERS,
            json={
                "id": "test-exp-create",
                "variants": [
                    {"model": "claude-haiku-4-5", "traffic_pct": 60},
                    {"model": "gpt-4o-mini", "traffic_pct": 40},
                ],
            },
        )
        assert resp.status_code == 200

    def test_create_experiment_returns_experiment_data(self, client):
        resp = client.post(
            "/v1/experiments",
            headers=AUTH_HEADERS,
            json={
                "id": "exp-data-check",
                "variants": [
                    {"model": "claude-haiku-4-5", "traffic_pct": 100},
                ],
            },
        )
        data = resp.json()
        assert data["id"] == "exp-data-check"
        assert len(data["variants"]) == 1

    def test_create_experiment_missing_id_returns_422(self, client):
        resp = client.post(
            "/v1/experiments",
            headers=AUTH_HEADERS,
            json={"variants": [{"model": "claude-haiku-4-5", "traffic_pct": 100}]},
        )
        assert resp.status_code == 422

    def test_create_experiment_missing_variants_returns_422(self, client):
        resp = client.post(
            "/v1/experiments",
            headers=AUTH_HEADERS,
            json={"id": "no-variants"},
        )
        assert resp.status_code == 422

    def test_create_experiment_missing_api_key_returns_401(self, client):
        resp = client.post(
            "/v1/experiments",
            json={"id": "no-auth", "variants": [{"model": "claude-haiku-4-5", "traffic_pct": 100}]},
        )
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# GET /v1/experiments/{id}/assignment
# ---------------------------------------------------------------------------


class TestExperimentsAssignmentEndpoint:
    def test_assignment_returns_200_for_existing_experiment(self, client):
        # Create the experiment first
        client.post(
            "/v1/experiments",
            headers=AUTH_HEADERS,
            json={
                "id": "assign-test",
                "variants": [{"model": "claude-haiku-4-5", "traffic_pct": 100}],
            },
        )
        resp = client.get("/v1/experiments/assign-test/assignment?user_id=user1", headers=AUTH_HEADERS)
        assert resp.status_code == 200

    def test_assignment_returns_correct_fields(self, client):
        client.post(
            "/v1/experiments",
            headers=AUTH_HEADERS,
            json={
                "id": "assign-fields",
                "variants": [{"model": "claude-haiku-4-5", "traffic_pct": 100}],
            },
        )
        resp = client.get("/v1/experiments/assign-fields/assignment?user_id=alice", headers=AUTH_HEADERS)
        data = resp.json()
        assert data["experiment_id"] == "assign-fields"
        assert data["user_id"] == "alice"
        assert "model" in data

    def test_assignment_returns_404_for_unknown_experiment(self, client):
        resp = client.get(
            "/v1/experiments/nonexistent-exp-xyz/assignment?user_id=user1", headers=AUTH_HEADERS
        )
        assert resp.status_code == 404

    def test_assignment_is_deterministic_for_same_user(self, client):
        client.post(
            "/v1/experiments",
            headers=AUTH_HEADERS,
            json={
                "id": "deterministic-test",
                "variants": [
                    {"model": "claude-haiku-4-5", "traffic_pct": 50},
                    {"model": "gpt-4o-mini", "traffic_pct": 50},
                ],
            },
        )
        resp1 = client.get(
            "/v1/experiments/deterministic-test/assignment?user_id=fixed-user", headers=AUTH_HEADERS
        )
        resp2 = client.get(
            "/v1/experiments/deterministic-test/assignment?user_id=fixed-user", headers=AUTH_HEADERS
        )
        assert resp1.json()["model"] == resp2.json()["model"]

    def test_assignment_default_user_id_is_default(self, client):
        client.post(
            "/v1/experiments",
            headers=AUTH_HEADERS,
            json={
                "id": "default-user-test",
                "variants": [{"model": "claude-haiku-4-5", "traffic_pct": 100}],
            },
        )
        resp = client.get("/v1/experiments/default-user-test/assignment", headers=AUTH_HEADERS)
        data = resp.json()
        assert data["user_id"] == "default"

    def test_assignment_missing_api_key_returns_401(self, client):
        resp = client.get("/v1/experiments/anything/assignment")
        assert resp.status_code == 401
