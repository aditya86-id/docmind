from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[3]
ENV_FILE = ROOT_DIR / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(ENV_FILE), extra="ignore")

    groq_api_key: str = Field(default="", validation_alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.1-8b-instant", validation_alias="GROQ_MODEL")
    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5", validation_alias="EMBEDDING_MODEL")
    chroma_dir: str = Field(default="storage/chroma", validation_alias="CHROMA_DIR")
    upload_dir: str = Field(default="storage/uploads", validation_alias="UPLOAD_DIR")
    cors_origins: str = Field(default="*", validation_alias="CORS_ORIGINS")
    host: str = Field(default="0.0.0.0", validation_alias="HOST")
    port: int = Field(default=8000, validation_alias="PORT")

    @property
    def chroma_path(self) -> Path:
        return Path(self.chroma_dir)

    @property
    def upload_path(self) -> Path:
        return Path(self.upload_dir)


@lru_cache
def get_settings() -> Settings:
    return Settings()
