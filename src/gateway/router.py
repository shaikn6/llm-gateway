"""Route requests to the correct LLM provider."""

from __future__ import annotations

from src.providers.anthropic import AnthropicProvider
from src.providers.openai_provider import OpenAIProvider


class GatewayRouter:
    def __init__(self, anthropic_key: str, openai_key: str = ""):
        self._anthropic = AnthropicProvider(api_key=anthropic_key)
        self._openai_key = openai_key
        self._openai = None

    def route(self, model: str):
        if model.startswith("gpt") or model.startswith("o1"):
            if self._openai is None:
                self._openai = OpenAIProvider(api_key=self._openai_key)
            return self._openai
        return self._anthropic
