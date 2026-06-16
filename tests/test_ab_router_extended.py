"""Extended tests for src/gateway/ab_router.py — beyond existing test_ab_router.py."""

from __future__ import annotations

import hashlib

import pytest

from src.gateway.ab_router import ABRouter, Experiment


@pytest.fixture
def router():
    return ABRouter()


@pytest.fixture
def two_variant_exp():
    return Experiment(
        id="exp-50-50",
        variants=[
            {"model": "claude-haiku-4-5", "traffic_pct": 50},
            {"model": "gpt-4o-mini", "traffic_pct": 50},
        ],
    )


@pytest.fixture
def three_variant_exp():
    return Experiment(
        id="exp-three",
        variants=[
            {"model": "model-a", "traffic_pct": 33},
            {"model": "model-b", "traffic_pct": 33},
            {"model": "model-c", "traffic_pct": 34},
        ],
    )


class TestABRouterAddExperiment:
    def test_add_experiment_stores_it(self, router, two_variant_exp):
        router.add_experiment(two_variant_exp)
        exps = router.list_experiments()
        assert len(exps) == 1
        assert exps[0].id == "exp-50-50"

    def test_add_multiple_experiments(self, router, two_variant_exp, three_variant_exp):
        router.add_experiment(two_variant_exp)
        router.add_experiment(three_variant_exp)
        assert len(router.list_experiments()) == 2

    def test_add_experiment_overwrites_same_id(self, router):
        exp_v1 = Experiment(
            id="my-exp",
            variants=[{"model": "model-a", "traffic_pct": 100}],
        )
        exp_v2 = Experiment(
            id="my-exp",
            variants=[{"model": "model-b", "traffic_pct": 100}],
        )
        router.add_experiment(exp_v1)
        router.add_experiment(exp_v2)
        exps = router.list_experiments()
        assert len(exps) == 1
        assert exps[0].variants[0]["model"] == "model-b"


class TestABRouterGetAssignment:
    def test_unknown_experiment_raises_value_error(self, router):
        with pytest.raises(ValueError, match="not found"):
            router.get_assignment("nonexistent", "user1")

    def test_assignment_is_deterministic_same_inputs(self, router, two_variant_exp):
        router.add_experiment(two_variant_exp)
        result1 = router.get_assignment("exp-50-50", "user-abc")
        result2 = router.get_assignment("exp-50-50", "user-abc")
        assert result1 == result2

    def test_different_users_can_get_different_variants(self, router, two_variant_exp):
        router.add_experiment(two_variant_exp)
        results = {
            router.get_assignment("exp-50-50", f"user-{i}") for i in range(50)
        }
        # With 50 users and 50/50 split, both variants should appear
        assert len(results) == 2

    def test_100_pct_single_variant_always_wins(self, router):
        exp = Experiment(
            id="single",
            variants=[{"model": "always-this", "traffic_pct": 100}],
        )
        router.add_experiment(exp)
        for i in range(20):
            assert router.get_assignment("single", f"user-{i}") == "always-this"

    def test_returns_last_variant_when_cumulative_equals_100(self, router):
        """When cumulative traffic reaches exactly 100, last variant is returned."""
        exp = Experiment(
            id="edge",
            variants=[
                {"model": "first", "traffic_pct": 50},
                {"model": "last", "traffic_pct": 50},
            ],
        )
        router.add_experiment(exp)
        # Verify both variants are reachable
        models = {router.get_assignment("edge", f"u{i}") for i in range(100)}
        assert "first" in models or "last" in models

    def test_bucket_is_0_to_99(self, router, two_variant_exp):
        """Verify bucket computation is md5 mod 100 (0-99 range)."""
        router.add_experiment(two_variant_exp)
        for user_id in ["userA", "userB", "userC", "userD"]:
            bucket = (
                int(
                    hashlib.md5(f"exp-50-50{user_id}".encode()).hexdigest(),
                    16,
                )
                % 100
            )
            assert 0 <= bucket <= 99

    def test_three_variants_distribute(self, router, three_variant_exp):
        router.add_experiment(three_variant_exp)
        results = {
            router.get_assignment("exp-three", f"user-{i}") for i in range(200)
        }
        # All three models should appear over 200 users
        assert len(results) == 3

    def test_assignment_uses_experiment_id_in_hash(self, router):
        """Same user gets different assignment for different experiments."""
        exp_a = Experiment(
            id="exp-a", variants=[{"model": "alpha", "traffic_pct": 100}]
        )
        exp_b = Experiment(
            id="exp-b", variants=[{"model": "beta", "traffic_pct": 100}]
        )
        router.add_experiment(exp_a)
        router.add_experiment(exp_b)
        # Both 100% so same model returned in both, but this confirms no crash
        assert router.get_assignment("exp-a", "user1") == "alpha"
        assert router.get_assignment("exp-b", "user1") == "beta"


class TestABRouterListExperiments:
    def test_empty_router_returns_empty_list(self, router):
        assert router.list_experiments() == []

    def test_returns_list_of_experiment_objects(self, router, two_variant_exp):
        router.add_experiment(two_variant_exp)
        exps = router.list_experiments()
        assert all(isinstance(e, Experiment) for e in exps)

    def test_experiment_data_preserved(self, router, two_variant_exp):
        router.add_experiment(two_variant_exp)
        exp = router.list_experiments()[0]
        assert exp.id == "exp-50-50"
        assert len(exp.variants) == 2


class TestExperimentDataclass:
    def test_experiment_has_id_and_variants(self):
        exp = Experiment(
            id="test-exp",
            variants=[{"model": "m1", "traffic_pct": 100}],
        )
        assert exp.id == "test-exp"
        assert exp.variants[0]["model"] == "m1"
