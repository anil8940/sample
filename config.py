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
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection: str = "documents-gemini"
    embedding_model: str = "gemini-embedding-2-preview"
    retrieval_k: int = 4
    max_upload_size_mb: int = 15


settings = Settings()
