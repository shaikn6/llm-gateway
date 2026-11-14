import builtins
from unittest.mock import Mock

import fakeredis
import pytest
from fastapi.testclient import TestClient

from src.api import deps, main
from src.cache.semantic_cache import SemanticCache
from src.config import settings
from tests.fakes import BODY, KEY_A, KEY_B, CountingProvider, make_cache


@pytest.fixture
def setup(monkeypatch):
    redis = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr("src.cache.semantic_cache.redis.from_url", lambda *a, **kw: redis)
    exact = SemanticCache()
    semantic = make_cache()
    provider = CountingProvider("fake")
    monkeypatch.setattr(main, "get_cache", lambda: exact)
    monkeypatch.setattr(main, "get_embedding_cache", lambda: semantic)
    monkeypatch.setattr(main, "get_router", lambda: Mock(route=lambda model: provider))
    tracker = Mock()
    monkeypatch.setattr(main, "get_usage_tracker", lambda: tracker)
    monkeypatch.setattr(deps, "get_rate_limiter", lambda: Mock(acquire=lambda key: (True, 100)))
    monkeypatch.setattr(settings, "semantic_cache_enabled", True)
    return TestClient(main.app), provider, semantic, exact, tracker


def post(client, text="Hello", headers=KEY_A):
    return client.post(
        "/v1/chat/completions",
        json={**BODY, "messages": [{"role": "user", "content": text}]},
        headers=headers,
    )


def test_semantic_route_and_invalidation(setup):
    client, provider, _semantic, _exact, tracker = setup
    assert post(client).json()["x-gateway-cache"] == "miss"
    assert post(client).json()["x-gateway-cache"] == "hit"
    response = post(client, "Hi")
    assert "x-gateway-cache" not in response.headers
    assert "x-gateway-cache-similarity" not in response.headers
    hit = response.json()
    assert hit["x-gateway-cache"] == "semantic-hit"
    assert hit["x-gateway-cache-similarity"] == 1.0
    assert hit["x-gateway-cost-usd"] == 0
    assert tracker.record.call_count == 1
    assert post(client, headers=KEY_B).json()["x-gateway-cache"] == "miss"
    assert client.delete("/v1/cache").status_code == 401
    assert client.delete("/v1/cache", headers=KEY_A).json() == {"exact": 1, "semantic": 1}
    assert post(client, headers=KEY_B).json()["x-gateway-cache"] == "hit"
    assert post(client).json()["x-gateway-cache"] == "miss"
    assert provider.calls == 3
    rendered = client.get("/metrics").text
    assert 'llm_gateway_semantic_cache_total{result="hit"} 1' in rendered
    assert 'llm_gateway_semantic_cache_total{result="miss"}' in rendered


def test_runtime_failure_and_disabled_delete(setup, monkeypatch):
    client, _provider, semantic, _exact, _tracker = setup
    semantic.embedder = Mock(side_effect=RuntimeError)
    assert post(client).status_code == 200
    assert "llm_gateway_semantic_cache_errors_total" in client.get("/metrics").text
    monkeypatch.setattr(settings, "semantic_cache_enabled", False)
    assert client.delete("/v1/cache", headers=KEY_A).json() == {"exact": 1, "semantic": 0}


def test_startup_missing_chroma(monkeypatch):
    real_import = builtins.__import__

    def missing(name, *args, **kwargs):
        if name.startswith("chromadb"):
            raise ImportError("missing chromadb")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing)
    monkeypatch.setattr(settings, "semantic_cache_enabled", True)
    main.get_embedding_cache.cache_clear()
    with pytest.raises(RuntimeError, match="Semantic cache startup failed"), TestClient(main.app):
        pass
    main.get_embedding_cache.cache_clear()


def test_factory_enabled_and_disabled(tmp_path, monkeypatch):
    pytest.importorskip("chromadb")
    main.get_embedding_cache.cache_clear()
    monkeypatch.setattr(settings, "semantic_cache_enabled", True)
    monkeypatch.setattr(settings, "semantic_cache_path", str(tmp_path))
    try:
        with TestClient(main.app):
            assert main.get_embedding_cache() is not None
    finally:
        main.get_embedding_cache.cache_clear()


def test_exact_invalidation_outage():
    import redis

    exact = SemanticCache()
    exact._redis = Mock()
    exact._redis.scan.side_effect = redis.RedisError
    assert exact.clear(tenant="a") == 0


def test_cache_version_invalidates_both_tiers(setup, monkeypatch):
    client, provider, _semantic, _exact, _tracker = setup
    post(client)
    monkeypatch.setattr(settings, "cache_version", "2")
    assert post(client).json()["x-gateway-cache"] == "miss"
    assert provider.calls == 2


def test_startup_embedder_failure(monkeypatch):
    pytest.importorskip("chromadb")
    monkeypatch.setattr(settings, "semantic_cache_enabled", True)
    monkeypatch.setattr(
        "chromadb.utils.embedding_functions.DefaultEmbeddingFunction",
        Mock(side_effect=RuntimeError("cannot load")),
    )
    main.get_embedding_cache.cache_clear()
    with pytest.raises(RuntimeError, match="cannot load"), TestClient(main.app):
        pass
    main.get_embedding_cache.cache_clear()


def test_disabled_startup_never_imports_chroma(monkeypatch):
    real_import = builtins.__import__

    def forbidden(name, *args, **kwargs):
        assert not name.startswith("chromadb")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", forbidden)
    monkeypatch.setattr(settings, "semantic_cache_enabled", False)
    main.get_embedding_cache.cache_clear()
    monkeypatch.setattr(main, "get_embedding_cache", Mock())
    try:
        with TestClient(main.app) as client:
            assert client.get("/health").status_code == 200
            main.get_embedding_cache.assert_not_called()
    finally:
        main.get_embedding_cache.cache_clear()


def test_disabled_completion_never_imports_optional_store(monkeypatch):
    import subprocess
    import sys

    script = """
import builtins
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    assert not name.startswith(("chromadb", "numpy", "src.cache.embedding_cache")), name
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
from fastapi.testclient import TestClient
from src.api.main import app
with TestClient(app) as client:
    assert client.get("/health").status_code == 200
"""
    monkeypatch.setenv("SEMANTIC_CACHE_ENABLED", "false")
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)


def test_disabled_exact_delete_and_store_error(setup, monkeypatch):
    client, _, semantic, exact, _ = setup
    monkeypatch.setattr(settings, "cache_enabled", False)
    exact.clear = Mock(side_effect=AssertionError("disabled exact cache"))
    assert client.delete("/v1/cache", headers=KEY_A).json() == {"exact": 0, "semantic": 0}
    semantic.store.evict = Mock(side_effect=RuntimeError("broken store"))
    semantic._writes = 99
    assert post(client).status_code == 200


def test_factory_preserves_store_error(monkeypatch):
    monkeypatch.setattr(
        "src.cache.embedding_cache.ChromaStore", Mock(side_effect=ValueError("bad path"))
    )
    main.get_embedding_cache.cache_clear()
    try:
        with pytest.raises(ValueError, match="bad path"):
            main.get_embedding_cache()
    finally:
        main.get_embedding_cache.cache_clear()


def test_exact_clear_batches_each_scan_page():
    exact = SemanticCache()
    exact._redis = Mock()
    exact._redis.scan.side_effect = [(7, ["first", "second"]), (0, ["third"])]
    exact._redis.unlink.side_effect = [2, 1]
    assert exact.clear(tenant="a") == 3
    assert [call.args for call in exact._redis.unlink.call_args_list] == [
        ("first", "second"),
        ("third",),
    ]
    assert all(
        call.kwargs["match"] == f"llm_cache:{exact._scope('a')}:*"
        for call in exact._redis.scan.call_args_list
    )
