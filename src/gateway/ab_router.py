"""A/B test router with deterministic assignment."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass
class Experiment:
    id: str
    variants: list[dict]  # [{model: str, traffic_pct: int}]


class ABRouter:
    def __init__(self):
        self._experiments: dict[str, Experiment] = {}

    def add_experiment(self, exp: Experiment) -> None:
        self._experiments[exp.id] = exp

    def get_assignment(self, experiment_id: str, user_id: str) -> str:
        exp = self._experiments.get(experiment_id)
        if not exp:
            raise ValueError(f"Experiment {experiment_id!r} not found")
        bucket = int(hashlib.md5(f"{experiment_id}{user_id}".encode()).hexdigest(), 16) % 100
        cumulative = 0
        for variant in exp.variants:
            cumulative += variant["traffic_pct"]
            if bucket < cumulative:
                return variant["model"]
        return exp.variants[-1]["model"]

    def list_experiments(self) -> list[Experiment]:
        return list(self._experiments.values())
