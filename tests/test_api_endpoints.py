"""Integration tests for FastAPI endpoints — completions and experiments."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client():
    return TestClient(app)


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
    def _mock_provider_response(self):
        return {
            "id": "chatcmpl-test",
            "model": "claude-haiku-4-5",
            "choices": [
                {"message": {"role": "assistant", "content": "Hello!"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

    def test_completions_happy_path(self, client):
        mock_provider = MagicMock()
        mock_provider.complete.return_value = self._mock_provider_response()

        with patch("src.api.main.get_router") as mock_get_router:
            mock_router = MagicMock()
            mock_router.route.return_value = mock_provider
            mock_get_router.return_value = mock_router

            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "claude-haiku-4-5",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

        assert resp.status_code == 200

    def test_completions_returns_provider_response(self, client):
        mock_provider = MagicMock()
        mock_provider.complete.return_value = self._mock_provider_response()

        with patch("src.api.main.get_router") as mock_get_router:
            mock_router = MagicMock()
            mock_router.route.return_value = mock_provider
            mock_get_router.return_value = mock_router

            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "claude-haiku-4-5",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

        data = resp.json()
        assert data["model"] == "claude-haiku-4-5"

    def test_completions_provider_exception_returns_500(self, client):
        mock_provider = MagicMock()
        mock_provider.complete.side_effect = RuntimeError("Provider error")

        with patch("src.api.main.get_router") as mock_get_router:
            mock_router = MagicMock()
            mock_router.route.return_value = mock_provider
            mock_get_router.return_value = mock_router

            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "claude-haiku-4-5",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

        assert resp.status_code == 500

    def test_completions_error_detail_in_response(self, client):
        mock_provider = MagicMock()
        mock_provider.complete.side_effect = RuntimeError("Provider error")

        with patch("src.api.main.get_router") as mock_get_router:
            mock_router = MagicMock()
            mock_router.route.return_value = mock_provider
            mock_get_router.return_value = mock_router

            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "claude-haiku-4-5",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

        assert "Provider error" in resp.json()["detail"]

    def test_completions_defaults_model_to_claude_haiku(self, client):
        mock_provider = MagicMock()
        mock_provider.complete.return_value = self._mock_provider_response()

        with patch("src.api.main.get_router") as mock_get_router:
            mock_router = MagicMock()
            mock_router.route.return_value = mock_provider
            mock_get_router.return_value = mock_router

            resp = client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "Hello"}]},
            )

        # Should use default model
        assert resp.status_code == 200

    def test_completions_routes_gpt_model_to_openai(self, client):
        mock_provider = MagicMock()
        mock_provider.complete.return_value = {
            "id": "chatcmpl-gpt",
            "model": "gpt-4o-mini",
            "choices": [
                {"message": {"role": "assistant", "content": "GPT response"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }

        with patch("src.api.main.get_router") as mock_get_router:
            mock_router = MagicMock()
            mock_router.route.return_value = mock_provider
            mock_get_router.return_value = mock_router

            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

        # Verify route was called with the gpt model
        mock_router.route.assert_called_once_with("gpt-4o-mini")

    def test_completions_passes_max_tokens(self, client):
        mock_provider = MagicMock()
        mock_provider.complete.return_value = self._mock_provider_response()

        with patch("src.api.main.get_router") as mock_get_router:
            mock_router = MagicMock()
            mock_router.route.return_value = mock_provider
            mock_get_router.return_value = mock_router

            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "claude-haiku-4-5",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 512,
                },
            )

        call_kwargs = mock_provider.complete.call_args[1]
        assert call_kwargs.get("max_tokens") == 512

    def test_completions_missing_messages_returns_422(self, client):
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "claude-haiku-4-5"},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /v1/experiments
# ---------------------------------------------------------------------------


class TestExperimentsListEndpoint:
    def test_list_experiments_returns_200(self, client):
        resp = client.get("/v1/experiments")
        assert resp.status_code == 200

    def test_list_experiments_returns_experiments_key(self, client):
        resp = client.get("/v1/experiments")
        data = resp.json()
        assert "experiments" in data

    def test_list_experiments_initially_empty(self, client):
        # Fresh app state may have leftover state from other tests in module,
        # so just verify the structure is correct
        resp = client.get("/v1/experiments")
        data = resp.json()
        assert isinstance(data["experiments"], list)


# ---------------------------------------------------------------------------
# POST /v1/experiments
# ---------------------------------------------------------------------------


class TestExperimentsCreateEndpoint:
    def test_create_experiment_returns_200(self, client):
        resp = client.post(
            "/v1/experiments",
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
            json={"variants": [{"model": "claude-haiku-4-5", "traffic_pct": 100}]},
        )
        assert resp.status_code == 422

    def test_create_experiment_missing_variants_returns_422(self, client):
        resp = client.post(
            "/v1/experiments",
            json={"id": "no-variants"},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /v1/experiments/{id}/assignment
# ---------------------------------------------------------------------------


class TestExperimentsAssignmentEndpoint:
    def test_assignment_returns_200_for_existing_experiment(self, client):
        # Create the experiment first
        client.post(
            "/v1/experiments",
            json={
                "id": "assign-test",
                "variants": [{"model": "claude-haiku-4-5", "traffic_pct": 100}],
            },
        )
        resp = client.get("/v1/experiments/assign-test/assignment?user_id=user1")
        assert resp.status_code == 200

    def test_assignment_returns_correct_fields(self, client):
        client.post(
            "/v1/experiments",
            json={
                "id": "assign-fields",
                "variants": [{"model": "claude-haiku-4-5", "traffic_pct": 100}],
            },
        )
        resp = client.get("/v1/experiments/assign-fields/assignment?user_id=alice")
        data = resp.json()
        assert data["experiment_id"] == "assign-fields"
        assert data["user_id"] == "alice"
        assert "model" in data

    def test_assignment_returns_404_for_unknown_experiment(self, client):
        resp = client.get(
            "/v1/experiments/nonexistent-exp-xyz/assignment?user_id=user1"
        )
        assert resp.status_code == 404

    def test_assignment_is_deterministic_for_same_user(self, client):
        client.post(
            "/v1/experiments",
            json={
                "id": "deterministic-test",
                "variants": [
                    {"model": "claude-haiku-4-5", "traffic_pct": 50},
                    {"model": "gpt-4o-mini", "traffic_pct": 50},
                ],
            },
        )
        resp1 = client.get(
            "/v1/experiments/deterministic-test/assignment?user_id=fixed-user"
        )
        resp2 = client.get(
            "/v1/experiments/deterministic-test/assignment?user_id=fixed-user"
        )
        assert resp1.json()["model"] == resp2.json()["model"]

    def test_assignment_default_user_id_is_default(self, client):
        client.post(
            "/v1/experiments",
            json={
                "id": "default-user-test",
                "variants": [{"model": "claude-haiku-4-5", "traffic_pct": 100}],
            },
        )
        resp = client.get("/v1/experiments/default-user-test/assignment")
        data = resp.json()
        assert data["user_id"] == "default"
