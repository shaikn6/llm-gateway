<p align="center"><img src=".github/banner.png" alt="llm-gateway" width="100%"></p>

<div align="center">

# LLM Gateway

[![CI](https://github.com/shaikn6/llm-gateway/actions/workflows/ci.yml/badge.svg)](https://github.com/shaikn6/llm-gateway/actions)
[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker)](docker-compose.yml)

**Production LLM gateway: OpenAI-compatible API + Redis caching + rate limiting + A/B testing — drop-in for direct LLM calls**

</div>

## Architecture

```mermaid
graph LR
  Client -->|X-API-Key| FastAPI
  FastAPI --> RateLimit[Rate Limiter\nRedis]
  FastAPI --> Cache[Semantic Cache\nRedis]
  FastAPI --> Router[Gateway Router]
  Router --> Anthropic[Anthropic\nClaude]
  Router --> OpenAI[OpenAI\nGPT-4o]
```

## Quick Start

```bash
git clone https://github.com/shaikn6/llm-gateway
cd llm-gateway && cp .env.example .env
docker compose up -d

curl http://localhost:8000/v1/chat/completions \
  -H "X-API-Key: dev-key-1" \
  -H "Content-Type: application/json" \
  -d '{"model": "claude-haiku-4-5", "messages": [{"role": "user", "content": "Hello"}]}'
```

## License
MIT

## API Reference

[![OpenAPI](https://img.shields.io/badge/OpenAPI-3.0-6BA539?logo=openapi-initiative&logoColor=white)](http://localhost:8000/docs)
[![Swagger UI](https://img.shields.io/badge/Swagger_UI-docs-85EA2D?logo=swagger&logoColor=black)](http://localhost:8000/docs)
[![ReDoc](https://img.shields.io/badge/ReDoc-redoc-8A2BE2)](http://localhost:8000/redoc)

Interactive docs: `http://localhost:8000/docs` (Swagger UI) · `http://localhost:8000/redoc` (ReDoc)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check — returns service version |
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat completions (routes to Anthropic or OpenAI) |
| `GET` | `/v1/experiments` | List all A/B experiments |
| `POST` | `/v1/experiments` | Create a new A/B experiment |
| `GET` | `/v1/experiments/{experiment_id}/assignment` | Get model assignment for a user |
