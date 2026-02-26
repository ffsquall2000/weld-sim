"""FastAPI dependency injection helpers."""
from __future__ import annotations

from web.config import WebConfig
from web.services.engine_service import EngineService

_engine_service: EngineService | None = None


def get_engine_service() -> EngineService:
    """Lazily create and return the singleton EngineService."""
    global _engine_service
    if _engine_service is None:
        _engine_service = EngineService.get_instance(data_dir=WebConfig.DATA_DIR)
        _engine_service.initialize()
    return _engine_service


def shutdown_engine_service() -> None:
    """Shut down the singleton EngineService and clear the module reference."""
    global _engine_service
    if _engine_service is not None:
        _engine_service.shutdown()
        _engine_service = None
