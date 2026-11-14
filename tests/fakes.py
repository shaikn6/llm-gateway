from src.cache.embedding_cache import EmbeddingCache
from src.models.schemas import ChatCompletionResponse, Choice, ChoiceMessage, UsageInfo


def _matches(metadata, where):
    if "$and" in where:
        return all(_matches(metadata, part) for part in where["$and"])
    for key, value in where.items():
        if isinstance(value, dict):
            op, bound = next(iter(value.items()))
            if not (metadata[key] >= bound if op == "$gte" else metadata[key] < bound):
                return False
        elif metadata[key] != value:
            return False
    return True


class InMemoryStore:
    """Small deterministic backend; numpy is optional until this backend is used."""

    def __init__(self):
        self.entries = {}

    def add(self, id, embedding, metadata):
        self.entries[id] = (embedding, metadata)

    def query(self, embedding, where):
        import numpy as np

        candidates = []
        for vector, metadata in self.entries.values():
            if _matches(metadata, where):
                norm = np.linalg.norm(embedding) * np.linalg.norm(vector)
                similarity = float(np.dot(embedding, vector) / norm) if norm else 0.0
                candidates.append((metadata, similarity))
        return max(candidates, key=lambda item: item[1]) if candidates else None

    def delete(self, where):
        ids = [id for id, (_, meta) in self.entries.items() if _matches(meta, where)]
        for id in ids:
            del self.entries[id]
        return len(ids)

    def evict(self, where, keep):
        cutoff = where["$and"][1]["created_at"]["$gte"]
        self.delete({"created_at": {"$lt": cutoff}})
        ids = sorted(
            (id for id, (_, meta) in self.entries.items() if _matches(meta, where)),
            key=lambda id: self.entries[id][1]["created_at"],
        )
        for id in ids[: max(0, len(ids) - keep)]:
            del self.entries[id]


KEY_A = {"X-API-Key": "dev-key-1"}
KEY_B = {"X-API-Key": "dev-key-2"}
BODY = {"model": "claude-haiku-4-5", "messages": [{"role": "user", "content": "Hello"}]}


class CountingProvider:
    """Stands in for an upstream; counts how often it is actually called."""

    def __init__(self, name: str):
        self.name = name
        self.calls = 0

    async def complete(self, request):
        self.calls += 1
        return ChatCompletionResponse(
            model=request.model,
            choices=[
                Choice(
                    index=0,
                    message=ChoiceMessage(content=f"answer #{self.calls} from {self.name}"),
                )
            ],
            usage=UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )


def make_cache(**kwargs):
    return EmbeddingCache(
        kwargs.pop("embedder", lambda texts: [[1.0, 0.0]]),
        kwargs.pop("store", InMemoryStore()),
        kwargs.pop("threshold", 0.95),
        kwargs.pop("ttl", 60),
        kwargs.pop("cap", 10),
    )
