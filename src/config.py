from functools import cached_property

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"
    redis_url: str = "redis://localhost:6379/0"
    cache_enabled: bool = True
    cache_ttl_s: int = 3600
    rate_limit_requests: int = 100
    rate_limit_window_s: int = 60
    api_keys: str = "dev-key-1,dev-key-2"
    log_level: str = "INFO"

    class Config:
        env_file = ".env"

    @cached_property
    def parsed_api_keys(self) -> set[str]:
        """Comma-separated api_keys config, parsed into a lookup set."""
        return {key.strip() for key in self.api_keys.split(",") if key.strip()}


settings = Settings()
