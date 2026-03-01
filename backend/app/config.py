"""Application configuration using pydantic-settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Database
    DATABASE_URL: str = (
        "postgresql+asyncpg://postgres:postgres@localhost:5432/weldsim"
    )

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # File storage
    STORAGE_PATH: str = "storage"

    # CORS
    CORS_ORIGINS: list[str] = ["*"]

    # Security
    SECRET_KEY: str = "dev-secret-key"

    # Debug mode
    DEBUG: bool = True

    @property
    def sync_database_url(self) -> str:
        """Return a synchronous database URL for Alembic migrations."""
        return self.DATABASE_URL.replace(
            "postgresql+asyncpg://", "postgresql+psycopg2://"
        )


settings = Settings()
