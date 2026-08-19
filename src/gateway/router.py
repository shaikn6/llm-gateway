"""Route requests to the correct LLM provider."""

from __future__ import annotations

from src.providers.anthropic import AnthropicProvider
from src.providers.ollama import OLLAMA_DEFAULT_URL, OllamaProvider
from src.providers.openai import OpenAIAsyncProvider as OpenAIProvider


class GatewayRouter:
    def __init__(
        self,
        anthropic_key: str,
        openai_key: str = "",
        ollama_base_url: str = OLLAMA_DEFAULT_URL,
    ):
        self._anthropic = AnthropicProvider(api_key=anthropic_key)
        self._openai_key = openai_key
        self._openai = None
        self._ollama_base_url = ollama_base_url
        self._ollama = None

    def route(self, model: str):
        # "ollama/<model>" namespacing (same convention LiteLLM and other
        # multi-provider gateways use for local models) -- explicit, so an
        # unrecognized model name still falls through to Anthropic below
        # rather than accidentally routing to a local Ollama instance.
        if model.startswith("ollama/"):
            if self._ollama is None:
                self._ollama = OllamaProvider(base_url=self._ollama_base_url)
            return self._ollama
        if model.startswith(("gpt", "o1")):
            if self._openai is None:
                self._openai = OpenAIProvider(api_key=self._openai_key)
            return self._openai
        return self._anthropic
