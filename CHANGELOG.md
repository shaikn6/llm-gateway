# Changelog

All notable changes to this project are documented here.

## [1.1.0] - 2026-10-10

Version numbers are now aligned: `pyproject.toml`, `src.__version__`, the
FastAPI app and `/health` all report `1.1.0` (previously `0.1.0` / `1.0.0`).

### Fixed
- Every Anthropic completion failed on a fresh install with
  `AsyncMessages.create() got an unexpected keyword argument 'temperature'`:
  `anthropic` 1.x removed `temperature` / `top_p` from `messages.create()` and
  the provider always sent both (as schema defaults). Sampling parameters are
  now sent only when the caller set them, via `extra_body` for Anthropic.
  The OpenAI and Ollama providers follow the same rule and now also forward
  `top_p` and `stop` (Ollama: as `options`).
- Response cache keyed only on `messages`: a cached answer could be served for
  a different model or different parameters, and to a different API key. The
  key now covers provider, model and every forwarded parameter and is scoped
  per API key.
- `messages: []` and malformed messages returned `500`; they are now `422`.
- Provider failures returned `500` with the raw exception text. They now map
  to `502` (upstream error) / `504` (upstream timeout) with a fixed message.
- Authenticated routes returned `500` when Redis was down. The rate limiter
  now fails closed with `503` + `Retry-After`, the cache degrades to a miss,
  and a failed usage write is dropped with a warning.

### Changed
- Rate-limit and usage entries are stored in Redis under an API-key fingerprint
  (`ratelimit:<sha256 prefix>`, `usage:<sha256 prefix>`) instead of the raw client key.
- **Breaking:** no default API keys. `API_KEYS` is required; the process
  refuses to start without it unless `GATEWAY_DEV_MODE=true`.
- **Breaking:** `/v1/chat/completions` rejects fields it does not forward
  (`stream: true`, `tools`, `tool_choice`, `response_format`, `n`, ...) with
  `422` instead of silently dropping them. `temperature`, `top_p` and `stop`
  are now accepted and forwarded.
- **Breaking:** `POST /v1/experiments` requires variants of the form
  `{model, traffic_pct}` whose `traffic_pct` values sum to 100 (`422` otherwise).
- `anthropic>=1.13,<2` and `openai>=3.28,<4` (previously unbounded `>=0.30` /
  `>=1.30`).
- Docs and descriptions call the cache what it is: an exact-match response
  cache, not a semantic one.
- Helm: `image.tag` defaults to the chart `appVersion` instead of `latest`;
  the chart now passes `API_KEYS` from its Secret; values document that no
  image is published.

### Added
- `tests/test_sdk_contract.py`: binds each provider's kwargs against the real
  installed SDK signatures and runs a completion through the real SDK request
  path against a fake transport.
- Dockerfile `HEALTHCHECK` on `/health`; compose Redis healthcheck and
  `depends_on: condition: service_healthy`.
- `scripts/fake_upstream.py`: standard-library stand-in for the provider APIs,
  for smoke-testing without keys.
- Structured JSON audit logging middleware (`src/middleware/audit.py`) — one
  metadata-only record per request (request id, method, path, status, latency,
  client IP, API-key fingerprint); prompt and completion content are never logged
- `Security` section in the README (threat model, encryption posture, audit
  logging, not-yet-covered gaps)

## [1.0.0] - 2026-06-16

### Added
- OpenAI-compatible REST API gateway with drop-in replacement support for existing SDK clients
- Redis-backed response cache keyed on a SHA-256 hash of the request payload, to avoid re-billing identical repeated requests
- Request routing across Claude, OpenAI, and Ollama backends
- Per-key rate limiting via a Redis sliding-window limiter
- Per-key usage tracking recorded as a background task after each response
- Docker Compose deployment with a Redis cache/rate-limit backend

### Changed
- Production-ready CI/CD with 95%+ test coverage enforcement

### Security
- All upstream API keys stored in environment variables; gateway never exposes backend credentials to clients
