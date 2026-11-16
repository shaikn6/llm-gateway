# Features

2026-11-14, working tree based on `454185c`: opt-in embedding cache, a local persistent store, tenant/request scope, numeric and contrast guards, expiry/caps, authenticated tenant invalidation, startup validation and semantic metrics. Configuration and CI extras remain optional for the base install. Bench scripts record threshold and replay measurements; [Semantic cache](SEMANTIC_CACHE.md) explains the results.

Validation: `.venv/bin/python -m pytest -q -p no:warnings --cov=src --cov-report=term` passed 466 tests with 99% displayed coverage (1,135/1,138 statements). `.venv/bin/ruff check .` and `.venv/bin/ruff format --check .` passed with 0.17.4. Existing-file formatting was normalized to satisfy the repository-wide format check.

Rollback: set `SEMANTIC_CACHE_ENABLED=false` and restart to bypass the second tier. `DELETE /v1/cache` clears the caller's entries; change `CACHE_VERSION` across replicas for namespace invalidation. The exact tier remains independently controlled by `CACHE_ENABLED`.

Both benchmark scripts ran locally and wrote JSON/PNG outputs under `bench/results/`. The shared threshold is 0.9997357130050659. The replay found no incremental semantic hits or savings; the held-out PAWS precision is 66.67%, below the 98% dev target.

2026-11-16, based on `58596c1`: documentation accuracy pass. The `pyproject.toml` comment said the SDKs were tested against anthropic 1.13.0 and capped at the next major, but a dependency bump had already widened the range to `anthropic>=1.13,<3`, so CI runs anthropic 2.x; the comment and the README threat-model row now state the real caps (`<3`, `<4`). The same row described the rate limiter as a pipelined round trip; it is one atomic Lua script. `docs/SEMANTIC_CACHE.md` now states where both benchmark inputs in `bench/data/` come from and that the 25-pair fintech test split is too small for tight estimates.

Validation: `pytest --cov=src --cov-report=term-missing --cov-fail-under=95` passed 476 tests at 100% line coverage; `ruff check src/ tests/` passed; the README quickstart was run from a fresh `.venv` and reproduced every row of its request table.

Rollback: revert the commit; no code paths changed.
