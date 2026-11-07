"""The Metrics registry and the unauthenticated GET /metrics scrape endpoint."""

from __future__ import annotations

import json
import re
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import src.api.deps as deps_module
import src.api.main as main_module
from src.api.main import app
from src.middleware.metrics import LATENCY_BUCKETS_MS, Metrics
from src.models.schemas import ChatCompletionResponse, Choice, ChoiceMessage, UsageInfo

VALID_API_KEY = "dev-key-1"
AUTH_HEADERS = {"X-API-Key": VALID_API_KEY}

# A metric identifier: `name` or `name{labels}`. Label values are quoted and may themselves contain
# braces (route templates like /v1/experiments/{experiment_id}/assignment), so the value is split off
# by the final space (Prometheus values never contain whitespace) rather than matched here.
_NAME = re.compile(r'^[a-zA-Z_:][a-zA-Z0-9_:]*(?:\{.*\})?$')


# ---------------------------------------------------------------------------
# Metrics registry (unit)
# ---------------------------------------------------------------------------


def _value(text, series):
    m = re.search(rf"^{re.escape(series)} ([0-9.e+-]+)$", text, re.MULTILINE)
    assert m, f"{series} not found in:\n{text}"
    return float(m.group(1))


def test_counters_accumulate_and_render_in_prometheus_format():
    m = Metrics()
    m.inc("things_total")
    m.inc("things_total", 2)
    m.inc("labelled_total", kind="a")
    text = m.render()
    assert "# TYPE things_total counter" in text
    assert _value(text, "things_total") == 3
    assert _value(text, 'labelled_total{kind="a"}') == 1


def test_histogram_buckets_are_cumulative_and_consistent():
    m = Metrics()
    for ms in (3, 7, 30, 20000):
        m.observe_request("/x", "GET", 200, ms)
    text = m.render()
    assert _value(text, 'llm_gateway_request_duration_ms_bucket{path="/x",le="5"}') == 1
    assert _value(text, 'llm_gateway_request_duration_ms_bucket{path="/x",le="10"}') == 2
    assert _value(text, 'llm_gateway_request_duration_ms_bucket{path="/x",le="50"}') == 3
    assert _value(text, f'llm_gateway_request_duration_ms_bucket{{path="/x",le="{LATENCY_BUCKETS_MS[-1]}"}}') == 3
    assert _value(text, 'llm_gateway_request_duration_ms_bucket{path="/x",le="+Inf"}') == 4
    assert _value(text, 'llm_gateway_request_duration_ms_count{path="/x"}') == 4
    assert _value(text, 'llm_gateway_requests_total{path="/x",method="GET",status="200"}') == 4


# ---------------------------------------------------------------------------
# GET /metrics (integration)
# ---------------------------------------------------------------------------


def _reset_singletons():
    main_module.get_router.cache_clear()
    main_module.get_cache.cache_clear()
    main_module.get_usage_tracker.cache_clear()
    deps_module.get_rate_limiter.cache_clear()


@pytest.fixture
def mock_redis():
    """Rate limiter always-allow, cache always-miss, backed by a mocked client so no live Redis is needed."""
    r = MagicMock()
    r.eval.return_value = [1, 99]
    r.get.return_value = None
    with (
        patch("src.middleware.rate_limiter.redis.from_url", return_value=r),
        patch("src.cache.semantic_cache.redis.from_url", return_value=r),
        patch("src.middleware.usage_tracker.redis.from_url", return_value=r),
    ):
        _reset_singletons()
        try:
            yield r
        finally:
            _reset_singletons()


@pytest.fixture
def client(mock_redis):
    return TestClient(app)


def _completion_response(model="claude-haiku-4-5"):
    return ChatCompletionResponse(
        id="chatcmpl-test",
        model=model,
        choices=[Choice(index=0, message=ChoiceMessage(role="assistant", content="hi"), finish_reason="stop")],
        usage=UsageInfo(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def _scrape(client):
    """Return the /metrics body and a {series: value} map, asserting every line parses as Prometheus text."""
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    parsed = {}
    for line in resp.text.splitlines():
        if not line or line.startswith("#"):
            continue
        series, _, value = line.rpartition(" ")
        assert _NAME.match(series), f"unparseable metrics line: {line!r}"
        parsed[series] = float(value)
    return resp.text, parsed


def test_metrics_endpoint_is_reachable_without_auth(client):
    # Reachable like /health: no X-API-Key, and not counted against any quota.
    assert client.get("/metrics").status_code == 200


def test_scrape_reflects_traffic_counters_after_requests(client, mock_redis):
    _, before = _scrape(client)

    # A cache hit (provider is never called) ...
    mock_redis.get.return_value = json.dumps(_completion_response().model_dump(by_alias=True))
    provider = AsyncMock()
    provider.name = "anthropic"
    with patch("src.api.main.get_router") as gr:
        gr.return_value.route.return_value = provider
        assert client.post("/v1/chat/completions", headers=AUTH_HEADERS,
                           json={"messages": [{"role": "user", "content": "x"}]}).status_code == 200
    provider.complete.assert_not_called()

    # ... a 422 (empty messages fails validation) ...
    assert client.post("/v1/chat/completions", headers=AUTH_HEADERS, json={"messages": []}).status_code == 422

    # ... and a 429 (rate limiter denies).
    mock_redis.eval.return_value = [0, 0]
    assert client.post("/v1/chat/completions", headers=AUTH_HEADERS,
                       json={"messages": [{"role": "user", "content": "x"}]}).status_code == 429

    text, after = _scrape(client)

    def delta(series):
        return after.get(series, 0.0) - before.get(series, 0.0)

    assert delta('llm_gateway_cache_total{result="hit"}') == 1
    assert delta('llm_gateway_requests_total{path="/v1/chat/completions",method="POST",status="422"}') == 1
    assert delta('llm_gateway_requests_total{path="/v1/chat/completions",method="POST",status="429"}') == 1
    assert delta("llm_gateway_rate_limited_total") == 1
    # The latency histogram for the route recorded all three calls.
    assert delta('llm_gateway_request_duration_ms_count{path="/v1/chat/completions"}') == 3
    assert "# TYPE llm_gateway_request_duration_ms histogram" in text


def test_scrape_reports_per_model_tokens_and_cost(client, mock_redis):
    _, before = _scrape(client)

    provider = AsyncMock()
    provider.name = "anthropic"
    provider.complete.return_value = _completion_response()
    with patch("src.api.main.get_router") as gr:
        gr.return_value.route.return_value = provider
        assert client.post("/v1/chat/completions", headers=AUTH_HEADERS,
                           json={"model": "claude-haiku-4-5", "messages": [{"role": "user", "content": "x"}]}).status_code == 200

    _, after = _scrape(client)

    def delta(series):
        return after.get(series, 0.0) - before.get(series, 0.0)

    assert delta('llm_gateway_cache_total{result="miss"}') == 1
    assert delta('llm_gateway_model_tokens_total{model="claude-haiku-4-5",type="prompt"}') == 10
    assert delta('llm_gateway_model_tokens_total{model="claude-haiku-4-5",type="completion"}') == 5
    # 10 prompt @ $0.25/1M + 5 completion @ $1.25/1M.
    expected = (10 * 0.25 + 5 * 1.25) / 1_000_000
    assert delta('llm_gateway_model_cost_usd_total{model="claude-haiku-4-5"}') == pytest.approx(expected)
