# Request flow

`create_completion` authenticates through `require_api_key`, resolves any experiment assignment, routes the selected provider, and tries the exact Redis cache. On an exact miss, the enabled embedding tier computes one vector and queries matching scope/guard/TTL metadata. A hit returns zero billed cost without usage accounting. Otherwise the provider completes, the enabled exact cache receives the response, and semantic writes and usage recording are scheduled as background tasks.

Enabled application lifespan calls `get_embedding_cache` in a worker thread and refuses startup if the embedder or persistent store cannot initialize. Runtime cache errors fall through. `DELETE /v1/cache` uses the completion auth dependency and clears only the caller's Redis prefix and local semantic tenant entries.
