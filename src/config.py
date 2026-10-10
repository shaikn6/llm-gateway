from functools import cached_property

from pydantic import model_validator
from pydantic_settings import BaseSettings

# Accepted only when GATEWAY_DEV_MODE=true and API_KEYS is unset. Never valid
# in a normally configured deployment.
DEV_API_KEYS = frozenset({"dev-key-1", "dev-key-2"})


class Settings(BaseSettings):
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"
    redis_url: str = "redis://localhost:6379/0"
    cache_enabled: bool = True
    cache_ttl_s: int = 3600
    rate_limit_requests: int = 100
    rate_limit_window_s: int = 60
    api_keys: str = ""
    gateway_dev_mode: bool = False
    log_level: str = "INFO"

    class Config:
        env_file = ".env"

    @model_validator(mode="after")
    def _require_api_keys(self) -> "Settings":
        """Refuse to start with no client keys rather than fall back to a guessable one."""
        if not self.parsed_api_keys:
            raise ValueError(
                "API_KEYS is not set. Set API_KEYS to a comma-separated list of client "
                "keys, or set GATEWAY_DEV_MODE=true to run locally with the built-in "
                "dev keys (never do this on a reachable host)."
            )
        return self

    @cached_property
    def parsed_api_keys(self) -> set[str]:
        """Comma-separated api_keys config, parsed into a lookup set."""
        keys = {key.strip() for key in self.api_keys.split(",") if key.strip()}
        if not keys and self.gateway_dev_mode:
            return set(DEV_API_KEYS)
        return keys


settings = Settings()
