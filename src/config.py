from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, populated from environment variables / .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    postgres_user: str
    postgres_password: str
    postgres_db: str
    postgres_host: str
    postgres_port: int

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


class MLSettings(BaseSettings):
    """Optional public ML snapshot settings, independent from database configuration."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ml_results_enabled: bool = False
    ml_export_manifest: Path | None = None
    ml_allow_unreliable: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]


@lru_cache
def get_ml_settings() -> MLSettings:
    return MLSettings()
