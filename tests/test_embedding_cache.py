import pytest

from src.cache.embedding_cache import EmbeddingCache, guard_keys
from tests.fakes import InMemoryStore, make_cache


def request(text="hello", **extra):
    return {
        "provider": "fake",
        "model": "test",
        "messages": [{"role": "user", "content": text}],
        **extra,
    }


def test_hit_and_vector_reuse():
    calls = []
    cache = EmbeddingCache(
        lambda texts: calls.append(texts) or [[1.0, 0.0]], InMemoryStore(), 0.95, 60, 10
    )
    hit, prepared = cache.lookup(request(), tenant="secret")
    assert hit is None
    cache.write(prepared, {"answer": "yes"})
    assert "secret" not in repr(cache.store.entries)
    assert "hello" not in repr(cache.store.entries)
    hit, _ = cache.lookup(request("hi"), tenant="secret")
    assert hit[0] == {"answer": "yes"}
    assert hit[1] == pytest.approx(1)
    assert len(calls) == 2


@pytest.mark.parametrize(
    "a,b", [("account 4421", "account 4422"), ("enable overdraft", "disable overdraft")]
)
def test_guards(a, b):
    assert guard_keys(a) != guard_keys(b)


def save(cache, req=None, tenant="a", response=None):
    _, prepared = cache.lookup(req or request(), tenant=tenant)
    cache.write(prepared, response or {"answer": "yes"})


@pytest.mark.parametrize(
    "change",
    [
        {"model": "other"},
        {"provider": "other"},
        {"temperature": 0.2},
        {
            "messages": [
                {"role": "system", "content": "different"},
                {"role": "user", "content": "hi"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "prior"},
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": "hi"},
            ]
        },
    ],
)
def test_scope(change):
    cache = make_cache()
    save(cache)
    assert cache.lookup(request(**change), tenant="a")[0] is None


def test_threshold_tenants_and_version():
    cache = make_cache(embedder=lambda texts: [[1.0, 0.0] if texts[0] == "hello" else [0.0, 1.0]])
    save(cache)
    assert cache.lookup(request("other"), tenant="a")[0] is None
    assert cache.lookup(request(), tenant="b")[0] is None
    assert cache.lookup(request(cache_version="2"), tenant="a")[0] is None
    assert cache.clear(tenant="b") == 0
    assert cache.clear(tenant="a") == 1


def test_expiry_and_cap(monkeypatch):
    import src.cache.embedding_cache as module

    now = [100.0]
    monkeypatch.setattr(module.time, "time", lambda: now[0])
    cache = make_cache(cap=1, ttl=10)
    save(cache, request("account 1"))
    now[0] += 1
    cache._writes = 99
    save(cache, request("account 2"))
    assert cache.lookup(request("account 1"), tenant="a")[0] is None
    assert cache.lookup(request("account 2"), tenant="a")[0] is not None
    now[0] += 11
    assert cache.lookup(request("account 2"), tenant="a")[0] is None
    cache._writes = 99
    save(cache)
    assert len(cache.store.entries) == 1


def test_errors_and_no_user(caplog):
    from unittest.mock import Mock

    from src.middleware.metrics import metrics

    before = metrics.render()
    cache = make_cache(embedder=Mock(side_effect=ValueError("private text")))
    assert cache.lookup(request(), tenant="a") == (None, None)
    cache.write(None, {})
    assert "private text" not in caplog.text
    assert "semantic cache operation failed" in caplog.text
    assert metrics.render() != before
    assert cache.lookup({"messages": []}, tenant="a") == (None, None)
    cache = make_cache(store=Mock())
    cache.store.query.side_effect = RuntimeError
    _, prepared = cache.lookup(request(), tenant="a")
    cache.store.delete.side_effect = RuntimeError
    cache.write(prepared, {})
    assert cache.clear(tenant="a") == 0


def test_zero_vector_and_guard_normalization():
    assert guard_keys("$5,000 12.5% no no") == guard_keys("5000 12.5 no")
    cache = make_cache(embedder=lambda texts: [[0.0, 0.0]])
    save(cache)
    assert cache.lookup(request(), tenant="a")[0] is None


def test_chroma_real(tmp_path):
    pytest.importorskip("chromadb")
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

    from src.cache.embedding_cache import ChromaStore

    store = ChromaStore(str(tmp_path))
    cache = make_cache(store=store, embedder=DefaultEmbeddingFunction(), cap=1)
    assert cache.lookup(request(), tenant="a")[0] is None
    save(cache)
    assert cache.lookup(request(), tenant="a")[0][1] > 0.99
    cache._writes = 99
    save(cache, request("account 2"))
    assert len(store.collection.get(include=[])["ids"]) == 1
    raw = store.collection.get(include=["documents", "metadatas"])
    assert raw["documents"] == ['{"answer": "yes"}']
    assert "hello" not in str(raw["metadatas"])
    assert cache.clear(tenant="a") == 1
    assert cache.clear(tenant="a") == 0


def test_cap_is_per_tenant_and_prior_context_stays_exact(monkeypatch):
    cache = make_cache(cap=1)
    save(cache, request("account 1"), tenant="b")
    save(cache, request("account 2"), tenant="a")
    cache._writes = 99
    save(cache, request("account 3"), tenant="a")
    assert cache.lookup(request("account 1"), tenant="b")[0] is not None
    assert cache.lookup(request("account 2"), tenant="a")[0] is None
    original = {
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "trailing answer"},
        ]
    }
    save(cache, request(**original))
    original["messages"][0]["content"] = "hi"
    assert cache.lookup(request(**original), tenant="a")[0] is not None
    original["messages"][1]["content"] = "changed trailing answer"
    assert cache.lookup(request(**original), tenant="a")[0] is None


def test_store_add_failure_is_nonfatal():
    from unittest.mock import Mock

    from src.middleware.metrics import metrics

    cache = make_cache()
    _, prepared = cache.lookup(request(), tenant="a")
    cache.store.add = Mock(side_effect=RuntimeError("private response"))
    before = metrics.render()
    cache.write(prepared, {"answer": "secret"})
    assert before != metrics.render()
    assert cache.lookup(request(), tenant="a")[0] is None


def test_housekeeping_cadence_and_lookup_do_not_mutate_request():
    from unittest.mock import Mock

    cache = make_cache()
    original = request()
    messages = original["messages"]
    last = messages[-1]
    cache.store.evict = Mock(wraps=cache.store.evict)
    for _ in range(99):
        save(cache, original)
    cache.store.evict.assert_not_called()
    save(cache, original)
    cache.store.evict.assert_called_once()
    assert original["messages"] is messages
    assert messages[-1] is last
    assert last["content"] == "hello"
    assert len(cache.store.entries) == 10


def test_chroma_housekeeping_only_fetches_ids_and_batches_deletes():
    from unittest.mock import Mock

    from src.cache.embedding_cache import ChromaStore

    store = ChromaStore.__new__(ChromaStore)
    store.collection = Mock()
    store.collection.get.side_effect = [{"ids": ["expired"]}, {"ids": ["003", "001", "002"]}]
    where = {"$and": [{"tenant": "a"}, {"created_at": {"$gte": 10}}]}
    store.evict(where, 1)
    assert all(call.kwargs["include"] == [] for call in store.collection.get.call_args_list)
    store.collection.delete.assert_called_once_with(ids=["expired", "001", "002"])


def test_lookup_add_and_clear_run_without_housekeeping_lock():
    class NoLock:
        def __enter__(self):
            raise RuntimeError("housekeeping lock")

        def __exit__(self, *args):
            pass

    cache = make_cache()
    cache._lock = NoLock()
    _, prepared = cache.lookup(request(), tenant="a")
    # Ordinary writes never acquire the housekeeping lock.
    cache.write(prepared, {"answer": "yes"})
    assert cache.lookup(request(), tenant="a")[0][0] == {"answer": "yes"}
    assert cache.clear(tenant="a") == 1
