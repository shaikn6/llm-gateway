# Features

2026-11-14, working tree based on `454185c`: opt-in embedding cache, a local persistent store, tenant/request scope, numeric and contrast guards, expiry/caps, authenticated tenant invalidation, startup validation and semantic metrics. Configuration and CI extras remain optional for the base install. Bench scripts record threshold and replay measurements; [Semantic cache](SEMANTIC_CACHE.md) explains the results.

Validation: `.venv/bin/python -m pytest -q -p no:warnings --cov=src --cov-report=term` passed 466 tests with 99% displayed coverage (1,135/1,138 statements). `.venv/bin/ruff check .` and `.venv/bin/ruff format --check .` passed with 0.17.4. Existing-file formatting was normalized to satisfy the repository-wide format check.

Rollback: set `SEMANTIC_CACHE_ENABLED=false` and restart to bypass the second tier. `DELETE /v1/cache` clears the caller's entries; change `CACHE_VERSION` across replicas for namespace invalidation. The exact tier remains independently controlled by `CACHE_ENABLED`.

Both benchmark scripts ran locally and wrote JSON/PNG outputs under `bench/results/`. The shared threshold is 0.9997357130050659. The replay found no incremental semantic hits or savings; the held-out PAWS precision is 66.67%, below the 98% dev target.
