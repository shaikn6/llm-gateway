"""Structured audit logging middleware."""

from __future__ import annotations

import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import src.api.deps as deps_module
import src.api.main as main_module
from src.api.main import app
from src.middleware.audit import JsonFormatter, key_fingerprint
from src.models.schemas import ChatCompletionResponse, Choice, ChoiceMessage, UsageInfo

VALID_API_KEY = "dev-key-1"
client = TestClient(app, headers={"X-API-Key": VALID_API_KEY})


def _reset_singletons():
    main_module.get_router.cache_clear()
    main_module.get_cache.cache_clear()
    main_module.get_usage_tracker.cache_clear()
    deps_module.get_rate_limiter.cache_clear()


@pytest.fixture(autouse=True)
def _mock_redis():
    """Back every Redis-using singleton with a mocked client (rate limit
    always-allow, cache always-miss) so these tests need no live server."""
    mock_redis = MagicMock()
    pipe = MagicMock()
    pipe.execute.return_value = [None, 0]
    mock_redis.pipeline.return_value = pipe
    mock_redis.get.return_value = None
    with (
        patch("src.middleware.rate_limiter.redis.from_url", return_value=mock_redis),
        patch("src.cache.semantic_cache.redis.from_url", return_value=mock_redis),
        patch("src.middleware.usage_tracker.redis.from_url", return_value=mock_redis),
    ):
        _reset_singletons()
        try:
            yield
        finally:
            _reset_singletons()


def _audit_record(caplog):
    return next(r for r in caplog.records if r.name == "llm_gateway.audit")


def test_audit_record_is_json_with_expected_fields(caplog):
    with caplog.at_level(logging.INFO, logger="llm_gateway.audit"):
        client.get("/health")
    data = json.loads(JsonFormatter().format(_audit_record(caplog)))
    for field in ("request_id", "method", "path", "status", "latency_ms"):
        assert field in data
    assert data["path"] == "/health"
    assert data["method"] == "GET"
    assert data["status"] == 200


def test_audit_reuses_inbound_request_id(caplog):
    with caplog.at_level(logging.INFO, logger="llm_gateway.audit"):
        client.get("/health", headers={"X-Request-Id": "abc-123"})
    data = json.loads(JsonFormatter().format(_audit_record(caplog)))
    assert data["request_id"] == "abc-123"


def test_audit_does_not_log_prompt_content(caplog):
    secret = "my-secret-prompt-text"
    mock_provider = AsyncMock()
    mock_provider.complete.return_value = ChatCompletionResponse(
        id="chatcmpl-test",
        model="claude-haiku-4-5",
        choices=[
            Choice(
                index=0,
                message=ChoiceMessage(role="assistant", content="ok"),
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )
    with caplog.at_level(logging.INFO, logger="llm_gateway.audit"), patch(
        "src.api.main.get_router"
    ) as gr:
        gr.return_value.route.return_value = mock_provider
        client.post(
            "/v1/chat/completions",
            json={"model": "claude-haiku-4-5", "messages": [{"role": "user", "content": secret}]},
        )
    assert any(r.name == "llm_gateway.audit" for r in caplog.records)
    for record in caplog.records:
        assert secret not in JsonFormatter().format(record)


def test_audit_logs_key_fingerprint_not_raw_key(caplog):
    with caplog.at_level(logging.INFO, logger="llm_gateway.audit"):
        client.get("/v1/experiments")
    data = json.loads(JsonFormatter().format(_audit_record(caplog)))
    assert data["api_key"] == key_fingerprint(VALID_API_KEY)
    assert data["api_key"] != VALID_API_KEY


def test_key_fingerprint_is_stable_short_and_distinct():
    assert key_fingerprint("dev-key-1") == key_fingerprint("dev-key-1")
    assert len(key_fingerprint("dev-key-1")) == 12
    assert key_fingerprint("dev-key-1") != key_fingerprint("dev-key-2")
