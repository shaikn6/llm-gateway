# Architecture

The FastAPI application in `src/api/main.py` routes authenticated completions through a provider router. Redis backs exact response caching, rate limiting and usage accounting. The opt-in `EmbeddingCache` uses a local Chroma collection of vectors and metadata; it never stores prompt documents. See [Semantic cache](SEMANTIC_CACHE.md) for its scope, guards, sensitivity and replica boundaries.
