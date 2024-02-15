from src.gateway.ab_router import ABRouter, Experiment


def test_deterministic():
    router = ABRouter()
    exp = Experiment(
        id="exp1",
        variants=[
            {"model": "claude-sonnet-4-6", "traffic_pct": 50},
            {"model": "gpt-4o", "traffic_pct": 50},
        ],
    )
    router.add_experiment(exp)
    assert router.get_assignment("exp1", "user1") == router.get_assignment("exp1", "user1")


def test_100pct_traffic():
    router = ABRouter()
    exp = Experiment(
        id="exp2",
        variants=[{"model": "claude-sonnet-4-6", "traffic_pct": 100}],
    )
    router.add_experiment(exp)
    assert router.get_assignment("exp2", "anyuser") == "claude-sonnet-4-6"
