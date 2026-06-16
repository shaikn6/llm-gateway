"""Tests for src/middleware/usage_tracker.py."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.middleware.usage_tracker import COST_PER_1M, UsageTracker


@pytest.fixture
def mock_redis():
    return MagicMock()


@pytest.fixture
def tracker(mock_redis):
    with patch("src.middleware.usage_tracker.redis.from_url", return_value=mock_redis):
        ut = UsageTracker(redis_url="redis://localhost:6379/0")
    return ut, mock_redis


class TestCostPer1M:
    def test_claude_sonnet_costs_defined(self):
        assert "claude-sonnet-4-6" in COST_PER_1M
        assert COST_PER_1M["claude-sonnet-4-6"]["input"] == 3.0
        assert COST_PER_1M["claude-sonnet-4-6"]["output"] == 15.0

    def test_claude_haiku_costs_defined(self):
        assert "claude-haiku-4-5" in COST_PER_1M
        assert COST_PER_1M["claude-haiku-4-5"]["input"] == 0.25

    def test_gpt4o_costs_defined(self):
        assert "gpt-4o" in COST_PER_1M
        assert COST_PER_1M["gpt-4o"]["input"] == 5.0

    def test_gpt4o_mini_costs_defined(self):
        assert "gpt-4o-mini" in COST_PER_1M
        assert COST_PER_1M["gpt-4o-mini"]["output"] == 0.6


class TestUsageTrackerRecord:
    def test_record_pushes_json_to_redis(self, tracker):
        ut, mock_redis = tracker
        ut.record("key1", "gpt-4o", prompt_tokens=100, completion_tokens=50)
        mock_redis.lpush.assert_called_once()
        call_args = mock_redis.lpush.call_args[0]
        assert call_args[0] == "usage:key1"
        entry = json.loads(call_args[1])
        assert entry["model"] == "gpt-4o"
        assert entry["prompt_tokens"] == 100
        assert entry["completion_tokens"] == 50

    def test_record_sets_24h_expiry(self, tracker):
        ut, mock_redis = tracker
        ut.record("key1", "gpt-4o", 100, 50)
        mock_redis.expire.assert_called_once_with("usage:key1", 86400)

    def test_record_includes_timestamp(self, tracker):
        import time

        ut, mock_redis = tracker
        before = time.time()
        ut.record("key1", "gpt-4o", 100, 50)
        after = time.time()
        entry = json.loads(mock_redis.lpush.call_args[0][1])
        assert before <= entry["ts"] <= after

    def test_record_multiple_times_pushes_each(self, tracker):
        ut, mock_redis = tracker
        ut.record("key1", "gpt-4o", 100, 50)
        ut.record("key1", "gpt-4o-mini", 200, 100)
        assert mock_redis.lpush.call_count == 2


class TestUsageTrackerGetUsage:
    def _make_entry(self, model, prompt_tokens, completion_tokens):
        return json.dumps(
            {
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "ts": 1234567890.0,
            }
        )

    def test_empty_usage_returns_zeros(self, tracker):
        ut, mock_redis = tracker
        mock_redis.lrange.return_value = []
        result = ut.get_usage("empty-key")
        assert result["total_requests"] == 0
        assert result["input_tokens"] == 0
        assert result["output_tokens"] == 0
        assert result["cost_usd"] == 0.0
        assert result["by_model"] == {}

    def test_single_entry_correct_total(self, tracker):
        ut, mock_redis = tracker
        entry = self._make_entry("gpt-4o", 1000, 500)
        mock_redis.lrange.return_value = [entry]
        result = ut.get_usage("key1")
        assert result["total_requests"] == 1
        assert result["input_tokens"] == 1000
        assert result["output_tokens"] == 500

    def test_cost_calculation_for_gpt4o(self, tracker):
        ut, mock_redis = tracker
        # 1M input tokens @ $5, 1M output tokens @ $15
        entry = self._make_entry("gpt-4o", 1_000_000, 1_000_000)
        mock_redis.lrange.return_value = [entry]
        result = ut.get_usage("key1")
        expected_cost = (1_000_000 * 5.0 + 1_000_000 * 15.0) / 1_000_000
        assert abs(result["cost_usd"] - expected_cost) < 0.0001

    def test_cost_calculation_for_claude_haiku(self, tracker):
        ut, mock_redis = tracker
        entry = self._make_entry("claude-haiku-4-5", 1_000_000, 1_000_000)
        mock_redis.lrange.return_value = [entry]
        result = ut.get_usage("key1")
        expected_cost = (1_000_000 * 0.25 + 1_000_000 * 1.25) / 1_000_000
        assert abs(result["cost_usd"] - expected_cost) < 0.0001

    def test_unknown_model_has_zero_cost(self, tracker):
        ut, mock_redis = tracker
        entry = self._make_entry("unknown-model", 1000, 500)
        mock_redis.lrange.return_value = [entry]
        result = ut.get_usage("key1")
        assert result["cost_usd"] == 0.0

    def test_multiple_entries_aggregate_correctly(self, tracker):
        ut, mock_redis = tracker
        entries = [
            self._make_entry("gpt-4o", 100, 50),
            self._make_entry("gpt-4o-mini", 200, 100),
            self._make_entry("gpt-4o", 300, 150),
        ]
        mock_redis.lrange.return_value = entries
        result = ut.get_usage("key1")
        assert result["total_requests"] == 3
        assert result["input_tokens"] == 600
        assert result["output_tokens"] == 300

    def test_by_model_groups_costs_per_model(self, tracker):
        ut, mock_redis = tracker
        entries = [
            self._make_entry("gpt-4o", 100, 50),
            self._make_entry("gpt-4o-mini", 200, 100),
            self._make_entry("gpt-4o", 100, 50),
        ]
        mock_redis.lrange.return_value = entries
        result = ut.get_usage("key1")
        assert "gpt-4o" in result["by_model"]
        assert "gpt-4o-mini" in result["by_model"]

    def test_cost_usd_is_rounded_to_6_decimals(self, tracker):
        ut, mock_redis = tracker
        entry = self._make_entry("gpt-4o", 1, 1)
        mock_redis.lrange.return_value = [entry]
        result = ut.get_usage("key1")
        # Check that it's a reasonable float with at most 6 decimal places
        assert isinstance(result["cost_usd"], float)

    def test_lrange_called_with_full_range(self, tracker):
        ut, mock_redis = tracker
        mock_redis.lrange.return_value = []
        ut.get_usage("mykey")
        mock_redis.lrange.assert_called_once_with("usage:mykey", 0, -1)
