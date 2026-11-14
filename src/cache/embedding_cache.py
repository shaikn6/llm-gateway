"""Opt-in guarded cosine cache. Response documents are fetched only on lookup."""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol
from uuid import uuid4

from src.cache.semantic_cache import digest
from src.middleware.audit import key_fingerprint
from src.middleware.metrics import metrics

logger = logging.getLogger(__name__)
# Heuristic measured in bench/: these tokens protect some semantic near-misses,
# not a proof of equivalence. Keep one shared lexicon for runtime and benchmarks.
CONTRAST_TOKENS = frozenset(
    [
        "not",
        "no",
        "never",
        "without",
        "disable",
        "enable",
        "remove",
        "add",
        "increase",
        "decrease",
        "in-network",
        "out-of-network",
        "before",
        "after",
        "receive",
        "receiving",
        "send",
        "sending",
        "joint",
        "individual",
    ]
)


def guard_keys(text: str) -> tuple[str, str]:
    # Deliberately no unit expansion (5k != 5000). Preserve multiplicity of numbers.
    numbers = re.findall(r"\d+(?:\.\d+)?", re.sub(r"[$,%]", "", text))
    tokens = set(re.findall(r"\b[\w]+(?:-[\w]+)*\b", text.lower()))
    return digest(sorted(numbers)), digest(sorted(tokens & CONTRAST_TOKENS))


class Store(Protocol):
    def add(self, id: str, embedding: list[float], metadata: dict) -> None: ...
    def query(self, embedding: list[float], where: dict) -> tuple[dict, float] | None: ...
    def delete(self, where: dict) -> int: ...
    def evict(self, where: dict, keep: int) -> None: ...


class ChromaStore:
    def __init__(self, path: str):
        import chromadb

        self.collection = chromadb.PersistentClient(path=path).get_or_create_collection(
            "gateway_responses",
            configuration={"hnsw": {"space": "cosine"}},
            embedding_function=None,
        )

    def add(self, id, embedding, metadata):
        metadata = dict(metadata)
        response = metadata.pop("response")
        self.collection.add(
            ids=[f"{metadata['created_at']:024.9f}:{id}"],
            embeddings=[embedding],
            metadatas=[metadata],
            documents=[response],
        )

    def query(self, embedding, where):
        result = self.collection.query(
            query_embeddings=[embedding],
            where=where,
            n_results=1,
            include=["metadatas", "distances", "documents"],
        )
        if not result["ids"][0]:
            return None
        return {**result["metadatas"][0][0], "response": result["documents"][0][0]}, 1 - result[
            "distances"
        ][0][0]

    def delete(self, where):
        ids = self.collection.get(where=where, include=[])["ids"]
        if ids:
            self.collection.delete(ids=ids)
        return len(ids)

    def evict(self, where, keep):
        # Timestamp-prefixed ids preserve creation order without loading responses
        # or metadata. TTL filters select expired entries across all tenants.
        cutoff = where["$and"][1]["created_at"]["$gte"]
        ids = self.collection.get(where={"created_at": {"$lt": cutoff}}, include=[])["ids"]
        current = sorted(self.collection.get(where=where, include=[])["ids"])
        ids.extend(current[: max(0, len(current) - keep)])
        if ids:
            self.collection.delete(ids=ids)


@dataclass
class Prepared:
    embedding: list[float]
    metadata: dict


class EmbeddingCache:
    def __init__(
        self,
        embedder: Callable,
        store: Store,
        threshold: float,
        ttl_s: int,
        max_entries_per_tenant: int,
    ):
        self.embedder = embedder
        self.store = store
        self.threshold = threshold
        self.ttl_s = ttl_s
        self.max_entries_per_tenant = max_entries_per_tenant
        self._writes = 0
        self._lock = threading.RLock()

    def warm(self):
        self.embedder(["startup check"])

    @staticmethod
    def _error(exc):
        logger.warning(
            "semantic cache operation failed (%s); treating as a miss", type(exc).__name__
        )
        metrics.inc("llm_gateway_semantic_cache_errors_total")

    def lookup(self, request: dict, *, tenant: str):
        """Returns (hit, prepared vector). The vector is reused for a subsequent write."""
        prepared = None
        try:
            scoped = {**request, "messages": list(request["messages"])}
            users = [i for i, m in enumerate(scoped["messages"]) if m["role"] == "user"]
            if not users:
                return None, None
            last = dict(scoped["messages"][users[-1]])
            text = last.pop("content")
            scoped["messages"][users[-1]] = last
            num, neg = guard_keys(text)
            fingerprint = key_fingerprint(tenant)
            metadata = {
                "scope": digest([fingerprint, scoped]),
                "num_key": num,
                "neg_key": neg,
                "tenant": fingerprint,
            }
            prepared = Prepared([float(v) for v in self.embedder([text])[0]], metadata)
            where = {
                "$and": [{k: v} for k, v in metadata.items()]
                + [{"created_at": {"$gte": time.time() - self.ttl_s}}]
            }
            result = self.store.query(prepared.embedding, where)
            if result and result[1] >= self.threshold:
                return (json.loads(result[0]["response"]), result[1]), prepared
        except Exception as exc:  # noqa: BLE001 - optional cache failures must not fail requests
            self._error(exc)
        return None, prepared

    def write(self, prepared: Prepared | None, response: dict):
        if prepared is None:
            return
        try:
            self.store.add(
                uuid4().hex,
                prepared.embedding,
                {**prepared.metadata, "created_at": time.time(), "response": json.dumps(response)},
            )
            # Approximate per-tenant cap: housekeeping runs every 100 writes.
            self._writes += 1
            if self._writes % 100 == 0:
                with self._lock:
                    self.store.evict(
                        {
                            "$and": [
                                {"tenant": prepared.metadata["tenant"]},
                                {"created_at": {"$gte": time.time() - self.ttl_s}},
                            ]
                        },
                        self.max_entries_per_tenant,
                    )
        except Exception as exc:  # noqa: BLE001 - optional cache failures must not fail requests
            self._error(exc)

    def clear(self, *, tenant: str) -> int:
        try:
            return self.store.delete({"tenant": key_fingerprint(tenant)})
        except Exception as exc:  # noqa: BLE001 - optional cache failures must not fail requests
            self._error(exc)
            return 0
