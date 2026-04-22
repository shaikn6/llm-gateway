# Changelog

All notable changes to this project are documented here.

## [1.0.0] - 2026-06-16

### Added
- OpenAI-compatible REST API gateway with drop-in replacement support for existing SDK clients
- Semantic caching layer using sentence-transformers to deduplicate near-identical prompts and cut costs
- Intelligent request routing across Claude, OpenAI, and Ollama backends based on latency and cost policy
- Per-model cost analytics dashboard with daily/weekly spend breakdowns and token-level attribution
- Rate limiting and quota enforcement per API key with configurable burst and sustained limits
- Docker Compose deployment with Redis cache backend and Prometheus metrics endpoint

### Changed
- Production-ready CI/CD with 95%+ test coverage enforcement

### Security
- All upstream API keys stored in environment variables; gateway never exposes backend credentials to clients
