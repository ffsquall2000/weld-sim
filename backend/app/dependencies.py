"""Application dependencies: async database engine, session, and Celery app."""

import logging
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from backend.app.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Async SQLAlchemy engine and session
# ---------------------------------------------------------------------------

# Build engine keyword arguments depending on the database backend.
# SQLite does not support connection-pool options like pool_pre_ping,
# pool_size, or max_overflow.
_engine_kwargs: dict = {}
if "postgresql" in settings.DATABASE_URL:
    _engine_kwargs.update(pool_pre_ping=True, pool_size=5, max_overflow=10)

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    **_engine_kwargs,
)

async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields an async database session."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ---------------------------------------------------------------------------
# Celery app (with Redis broker) – gracefully degrade when Redis is absent
# ---------------------------------------------------------------------------

try:
    from celery import Celery

    celery_app = Celery(
        "weldsim",
        broker=settings.REDIS_URL,
        backend=settings.REDIS_URL,
        include=[
            "backend.app.tasks.solver_tasks",
            "backend.app.tasks.optimization_tasks",
        ],
    )

    celery_app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
        task_track_started=True,
        task_acks_late=True,
        worker_prefetch_multiplier=1,
    )
except Exception:
    logger.warning(
        "Celery/Redis not available – background tasks will be disabled"
    )
    celery_app = None  # type: ignore[assignment]
