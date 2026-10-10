"""Shared pytest fixtures for llm-gateway tests."""

from __future__ import annotations

import os

# src.config refuses to load without API keys. Set them before anything under
# src/ is imported; real environment variables also outrank a developer's .env.
os.environ["API_KEYS"] = "dev-key-1,dev-key-2"
os.environ.pop("GATEWAY_DEV_MODE", None)

import pytest
from fastapi.testclient import TestClient

from src.models.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    ChoiceMessage,
    MessageRole,
    UsageInfo,
)


@pytest.fixture
def sample_messages():
    return [{"role": "user", "content": "Hello"}]


@pytest.fixture
def sample_chat_request():
    """A minimal valid ChatCompletionRequest."""
    return ChatCompletionRequest(
        model="claude-haiku-4-5",
        messages=[ChatMessage(role=MessageRole.user, content="Hello")],
    )


@pytest.fixture
def sample_chat_response():
    """A minimal valid ChatCompletionResponse."""
    return ChatCompletionResponse(
        model="claude-haiku-4-5",
        choices=[
            Choice(
                index=0,
                message=ChoiceMessage(role="assistant", content="Hi there!"),
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(prompt_tokens=5, completion_tokens=3, total_tokens=8),
    )


@pytest.fixture
def api_client():
    """FastAPI TestClient for integration tests."""
    from src.api.main import app
    return TestClient(app)
