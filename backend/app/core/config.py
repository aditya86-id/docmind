from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    embedding_model: str = "all-MiniLM-L6-v2"  # HuggingFace local model (Groq has no embedding API)
    chroma_dir: str = "storage/chroma"
    upload_dir: str = "storage/uploads"
    cors_origins: str = "*"
    host: str = "0.0.0.0"
    port: int = 8000

    @property
    def chroma_path(self) -> Path:
        return Path(self.chroma_dir)

    @property
    def upload_path(self) -> Path:
        return Path(self.upload_dir)


@lru_cache
def get_settings() -> Settings:
    return Settings()
