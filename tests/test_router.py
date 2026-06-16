from src.gateway.router import GatewayRouter
from src.providers.anthropic import AnthropicProvider


def test_claude_routes_to_anthropic():
    router = GatewayRouter(anthropic_key="test", openai_key="test")
    provider = router.route("claude-sonnet-4-6")
    assert isinstance(provider, AnthropicProvider)


def test_unknown_defaults_to_anthropic():
    router = GatewayRouter(anthropic_key="test", openai_key="test")
    provider = router.route("unknown-model")
    assert isinstance(provider, AnthropicProvider)
