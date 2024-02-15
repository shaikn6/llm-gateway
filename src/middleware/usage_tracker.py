"""Track token usage and cost per API key."""

from __future__ import annotations

import json
import time

import redis

COST_PER_1M = {
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5": {"input": 0.25, "output": 1.25},
    "gpt-4o": {"input": 5.0, "output": 15.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.6},
}


class UsageTracker:
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self._redis = redis.from_url(redis_url, decode_responses=True)

    def record(self, api_key: str, model: str, prompt_tokens: int, completion_tokens: int) -> None:
        entry = json.dumps(
            {
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "ts": time.time(),
            }
        )
        self._redis.lpush(f"usage:{api_key}", entry)
        self._redis.expire(f"usage:{api_key}", 86400)

    def get_usage(self, api_key: str) -> dict:
        entries = self._redis.lrange(f"usage:{api_key}", 0, -1)
        total_input = total_output = total_cost = 0
        by_model: dict = {}
        for e in entries:
            d = json.loads(e)
            m = d["model"]
            total_input += d["prompt_tokens"]
            total_output += d["completion_tokens"]
            cost_map = COST_PER_1M.get(m, {"input": 0, "output": 0})
            cost = (
                d["prompt_tokens"] * cost_map["input"] + d["completion_tokens"] * cost_map["output"]
            ) / 1_000_000
            total_cost += cost
            by_model[m] = by_model.get(m, 0) + cost
        return {
            "total_requests": len(entries),
            "input_tokens": total_input,
            "output_tokens": total_output,
            "cost_usd": round(total_cost, 6),
            "by_model": by_model,
        }
