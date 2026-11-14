"""In-process replay. Hit rate depends on the workload; this is not a production forecast."""

import asyncio
import json
import os
import tempfile
import time
from collections import Counter
from pathlib import Path
from unittest.mock import Mock, patch

os.environ.setdefault("API_KEYS", "bench-key")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/semantic-matplotlib")
import fakeredis
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from fastapi.testclient import TestClient

from bench.tune_threshold import rows
from src.api import deps, main
from src.cache.embedding_cache import ChromaStore, EmbeddingCache
from src.cache.semantic_cache import SemanticCache
from src.config import settings
from src.middleware.usage_tracker import request_cost_usd
from src.models.schemas import ChatCompletionResponse, Choice, ChoiceMessage, UsageInfo

ROOT = Path(__file__).resolve().parent
MODEL = "gpt-4o-mini"


class Provider:
    name = "fake"

    def __init__(self, latency):
        self.latency = latency
        self.calls = 0

    async def complete(self, request):
        await asyncio.sleep(self.latency / 1000)
        self.calls += 1
        return ChatCompletionResponse(
            model=MODEL,
            choices=[Choice(index=0, message=ChoiceMessage(content=f"response {self.calls}"))],
            usage=UsageInfo(prompt_tokens=100, completion_tokens=50, total_tokens=150),
        )


def percentiles(values):
    return {
        "count": len(values),
        "p50_ms": float(np.percentile(values, 50)) if values else None,
        "p95_ms": float(np.percentile(values, 95)) if values else None,
    }


def main_run():
    latency = float(os.getenv("FAKE_UPSTREAM_LATENCY_MS", "200"))
    threshold = json.loads((ROOT / "results/threshold.json").read_text())["threshold"]
    pairs = [("fintech", row) for row in rows("fintech_pairs")] + [
        ("paws", row) for row in rows("paws_sample")[:300]
    ]
    report = {
        "note": "Hit rate depends on the workload. Pairs isolated by exact system scope; each sends text_a then text_b. No additional exact-repeat traffic.",
        "pairs": len(pairs),
        "requests": len(pairs) * 2,
        "provider_latency_ms": latency,
        "threshold": threshold,
        "model": MODEL,
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "configs": {},
    }
    embed = DefaultEmbeddingFunction()
    embed(["warmup"])
    for config in ["no_cache", "exact_only", "exact_semantic"]:
        with tempfile.TemporaryDirectory() as directory:
            fake = fakeredis.FakeRedis(decode_responses=True)
            provider = Provider(latency)
            exact = SemanticCache()
            exact._redis = fake
            semantic = EmbeddingCache(embed, ChromaStore(directory), threshold, 3600, 10000)
            counts = Counter()
            hits = []
            misses = []
            false_hits = 0
            with (
                patch.object(settings, "cache_enabled", config != "no_cache"),
                patch.object(settings, "semantic_cache_enabled", config == "exact_semantic"),
                patch.object(main, "get_cache", return_value=exact),
                patch.object(main, "get_embedding_cache", return_value=semantic),
                patch.object(
                    main,
                    "get_router",
                    return_value=Mock(route=lambda model, provider=provider: provider),
                ),
                patch.object(main, "get_usage_tracker", return_value=Mock()),
                patch.object(
                    deps, "get_rate_limiter", return_value=Mock(acquire=lambda key: (True, 100000))
                ),
                TestClient(main.app) as client,
            ):
                for index, (dataset, row) in enumerate(pairs):
                    for side in ["text_a", "text_b"]:
                        started = time.perf_counter()
                        response = client.post(
                            "/v1/chat/completions",
                            headers={"X-API-Key": "bench-key"},
                            json={
                                "model": MODEL,
                                "messages": [
                                    {"role": "system", "content": f"pair {dataset} {index}"},
                                    {"role": "user", "content": row[side]},
                                ],
                            },
                        )
                        elapsed = (time.perf_counter() - started) * 1000
                        response.raise_for_status()
                        kind = response.json()["x-gateway-cache"]
                        counts[kind] += 1
                        (hits if kind in {"hit", "semantic-hit"} else misses).append(elapsed)
                        if (
                            side == "text_b"
                            and row["same_meaning"] == "0"
                            and kind == "semantic-hit"
                        ):
                            false_hits += 1
                    if index % 50 == 0:
                        print(config, index, "/", len(pairs), flush=True)
            total = len(pairs) * 2
            avoided = total - provider.calls
            report["configs"][config] = {
                "provider_calls": provider.calls,
                "exact_hits": counts["hit"],
                "semantic_hits": counts["semantic-hit"],
                "exact_hit_rate": counts["hit"] / total,
                "semantic_hit_rate": counts["semantic-hit"] / total,
                "false_semantic_hits": false_hits,
                "dollars_saved": avoided * request_cost_usd(MODEL, 100, 50),
                "provider_cost_usd": provider.calls * request_cost_usd(MODEL, 100, 50),
                "hits": percentiles(hits),
                "misses": percentiles(misses),
            }
    report["incremental_semantic_savings_usd"] = (
        report["configs"]["exact_semantic"]["dollars_saved"]
        - report["configs"]["exact_only"]["dollars_saved"]
    )
    (ROOT / "results/cost_latency.json").write_text(json.dumps(report, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    names = list(report["configs"])
    axes[0].bar(names, [report["configs"][n]["exact_hit_rate"] for n in names], label="exact")
    axes[0].bar(
        names,
        [report["configs"][n]["semantic_hit_rate"] for n in names],
        bottom=[report["configs"][n]["exact_hit_rate"] for n in names],
        label="semantic",
    )
    axes[0].set(title="Hit rate depends on this workload", ylabel="fraction of requests")
    axes[0].legend()
    x = np.arange(len(names))
    axes[1].bar(
        x - 0.18,
        [report["configs"][n]["misses"]["p50_ms"] or 0 for n in names],
        width=0.36,
        label="miss p50",
    )
    axes[1].bar(
        x + 0.18,
        [report["configs"][n]["hits"]["p50_ms"] or 0 for n in names],
        width=0.36,
        label="hit p50 (0 = none)",
    )
    axes[1].set(
        xticks=x, xticklabels=names, ylabel="milliseconds", title=f"Provider latency {latency:g} ms"
    )
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(ROOT / "results/cost_latency.png", dpi=160)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main_run()
