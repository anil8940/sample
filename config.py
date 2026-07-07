"""Configuration settings for the application."""

from pydantic_settings import BaseSettings
from pydantic import ConfigDict


class Settings(BaseSettings):
    """Application settings."""
    
    model_config = ConfigDict(env_file=".env", extra="ignore")
    
    app_title: str = "LLM API"
    app_description: str = "Simple FastAPI wrapper for LLM"
    debug: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    openrouter_api_key: str = ""


settings = Settings()
