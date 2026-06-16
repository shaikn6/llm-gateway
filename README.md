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
