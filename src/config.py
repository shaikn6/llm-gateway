from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    redis_url: str = "redis://localhost:6379/0"
    cache_enabled: bool = True
    cache_ttl_s: int = 3600
    rate_limit_requests: int = 100
    rate_limit_window_s: int = 60
    api_keys: str = "dev-key-1,dev-key-2"
    log_level: str = "INFO"

    class Config:
        env_file = ".env"


settings = Settings()
