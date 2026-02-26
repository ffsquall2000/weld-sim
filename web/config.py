"""Web application configuration via environment variables."""
from __future__ import annotations

import os


class WebConfig:
    """Configuration for the FastAPI web service.

    All values are read from environment variables with sensible defaults.
    """

    HOST: str = os.environ.get("UWM_HOST", "0.0.0.0")
    PORT: int = int(os.environ.get("UWM_PORT", "8001"))
    DATA_DIR: str = os.environ.get("UWM_DATA_DIR", "data")
    CORS_ORIGINS: list[str] = [
        origin.strip()
        for origin in os.environ.get("UWM_CORS_ORIGINS", "*").split(",")
        if origin.strip()
    ]
    REPORTS_DIR: str = os.environ.get("UWM_REPORTS_DIR", "reports")
    UPLOADS_DIR: str = os.environ.get("UWM_UPLOADS_DIR", "uploads")
