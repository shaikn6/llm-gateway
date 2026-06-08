"""OpenAI provider (simple sync interface for gateway routing)."""

from __future__ import annotations

import openai


class OpenAIProvider:
    """Lightweight OpenAI wrapper used by GatewayRouter."""

    def __init__(self, api_key: str):
        self._client = openai.OpenAI(api_key=api_key)

    def complete(self, messages: list[dict], model: str = "gpt-4o-mini", **kwargs) -> dict:
        resp = self._client.chat.completions.create(model=model, messages=messages, **kwargs)
        return {
            "id": resp.id,
            "model": resp.model,
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": resp.choices[0].message.content,
                    }
                }
            ],
            "usage": {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
            },
        }
