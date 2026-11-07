"""Minimal Prometheus-format metrics (counters and a latency histogram), thread-safe, no external dependency."""

from __future__ import annotations

import threading
from bisect import bisect_left
from collections import defaultdict

LATENCY_BUCKETS_MS = (5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000)
REQUESTS = "llm_gateway_requests_total"
DURATION = "llm_gateway_request_duration_ms"


def _fmt(labels):
    return ",".join(f'{k}="{v}"' for k, v in labels)


class Metrics:
    """Label values must come from code (route templates, status codes, model names), never from raw user
    input, so cardinality stays bounded.

    The request path only touches a dict key and a few numbers under the lock; label strings and cumulative
    bucket counts are built when the metrics are scraped.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._counters = defaultdict(float)  # (name, labels tuple) -> value
        self._requests = defaultdict(int)  # (path, method, status) -> count
        self._latency = {}  # path -> [per-bucket counts (last slot = overflow), sum_ms, count]

    def inc(self, name, amount=1.0, **labels):
        with self._lock:
            self._counters[(name, tuple(labels.items()))] += amount

    def observe_request(self, path, method, status, ms):
        slot = bisect_left(LATENCY_BUCKETS_MS, ms)
        with self._lock:
            self._requests[(path, method, status)] += 1
            h = self._latency.setdefault(path, [[0] * (len(LATENCY_BUCKETS_MS) + 1), 0.0, 0])
            h[0][slot] += 1
            h[1] += ms
            h[2] += 1

    def render(self):
        with self._lock:
            counters, requests = dict(self._counters), dict(self._requests)
            latency = {p: ([*h[0]], h[1], h[2]) for p, h in self._latency.items()}
        series = [(n, labels, v) for (n, labels), v in counters.items()]
        series += [(REQUESTS, (("path", p), ("method", m), ("status", s)), v) for (p, m, s), v in requests.items()]
        lines, current = [], None
        for name, labels, value in sorted(series):
            if name != current:
                current = name
                lines.append(f"# TYPE {name} counter")
            lines.append(f"{name}{{{_fmt(labels)}}} {value:g}" if labels else f"{name} {value:g}")
        if latency:
            lines.append(f"# TYPE {DURATION} histogram")
        for path, (counts, total_ms, count) in sorted(latency.items()):
            running = 0
            for upper, c in zip(LATENCY_BUCKETS_MS, counts):
                running += c
                lines.append(f'{DURATION}_bucket{{path="{path}",le="{upper}"}} {running}')
            lines.append(f'{DURATION}_bucket{{path="{path}",le="+Inf"}} {count}')
            lines.append(f'{DURATION}_sum{{path="{path}"}} {total_ms:.1f}')
            lines.append(f'{DURATION}_count{{path="{path}"}} {count}')
        return "\n".join(lines) + "\n"


# Process-wide instance the app and middleware share (mirrors the other singletons in src/api/main.py).
metrics = Metrics()
