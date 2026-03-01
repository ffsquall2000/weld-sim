"""FastAPI application factory for the Ultrasonic Metal Welding Virtual Simulation Platform."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.app.config import settings
from backend.app.routers import (
    comparisons,
    geometries,
    health,
    materials,
    optimizations,
    projects,
    runs,
    simulations,
    workflows,
    ws,
)

logger = logging.getLogger(__name__)

# Path to the frontend build output (Vite dist)
_FRONTEND_DIST = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: startup and shutdown hooks."""
    # --- Startup ---
    logger.info("WeldSim v2 starting up ...")
    # TODO: initialize async DB engine / session factory
    # TODO: initialize Redis connection pool
    # TODO: initialize Celery / task-queue worker connections
    yield
    # --- Shutdown ---
    logger.info("WeldSim v2 shutting down ...")
    # TODO: close DB engine
    # TODO: close Redis pool


def create_app() -> FastAPI:
    """Build and return the configured FastAPI application."""

    app = FastAPI(
        title="Ultrasonic Metal Welding Virtual Simulation Platform",
        description=(
            "REST + WebSocket API for parametric horn geometry, FEA simulation, "
            "optimization studies, and result comparison."
        ),
        version="2.0.0",
        lifespan=lifespan,
        docs_url="/api/v2/docs",
        redoc_url="/api/v2/redoc",
        openapi_url="/api/v2/openapi.json",
    )

    # ---- CORS ----
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---- Exception handlers ----
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"detail": str(exc)},
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.exception("Unhandled exception: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )

    # ---- API v2 routers ----
    api_prefix = "/api/v2"

    app.include_router(health.router, prefix=api_prefix)
    app.include_router(projects.router, prefix=api_prefix)
    app.include_router(geometries.router, prefix=api_prefix)
    app.include_router(simulations.router, prefix=api_prefix)
    app.include_router(runs.router, prefix=api_prefix)
    app.include_router(materials.router, prefix=api_prefix)
    app.include_router(comparisons.router, prefix=api_prefix)
    app.include_router(optimizations.router, prefix=api_prefix)
    app.include_router(workflows.router, prefix=api_prefix)
    app.include_router(ws.router, prefix=api_prefix)

    # ---- Static file serving for SPA frontend ----
    if _FRONTEND_DIST.is_dir():
        app.mount(
            "/",
            StaticFiles(directory=str(_FRONTEND_DIST), html=True),
            name="frontend",
        )

    return app


# Module-level app instance for ``uvicorn backend.app.main:app``
app = create_app()
