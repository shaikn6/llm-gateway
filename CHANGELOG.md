# Changelog

All notable changes to this project are documented here.

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
