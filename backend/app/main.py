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
    manual_optimization,
    materials,
    optimizations,
    projects,
    reports,
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
    app.include_router(reports.router, prefix=api_prefix)
    app.include_router(optimizations.router, prefix=api_prefix)
    app.include_router(workflows.router, prefix=api_prefix)
    app.include_router(ws.router, prefix=api_prefix)
    app.include_router(manual_optimization.router, prefix=api_prefix)

    # ---- Mount legacy v1 API routers ----
    try:
        from web.routers import (
            health as v1_health,
            calculation, materials as v1_materials, recipes, reports,
            geometry, horn, acoustic, knurl, suggestions, assembly,
            ws as v1_ws, mesh_data,
        )
        from web.dependencies import get_engine_service, shutdown_engine_service

        v1_prefix = "/api/v1"
        app.include_router(v1_health.router, prefix=v1_prefix)
        app.include_router(calculation.router, prefix=v1_prefix)
        app.include_router(v1_materials.router, prefix=v1_prefix)
        app.include_router(recipes.router, prefix=v1_prefix)
        app.include_router(reports.router, prefix=v1_prefix)
        app.include_router(geometry.router, prefix=v1_prefix)
        app.include_router(horn.router, prefix=v1_prefix)
        app.include_router(acoustic.router, prefix=v1_prefix)
        app.include_router(knurl.router, prefix=v1_prefix)
        app.include_router(suggestions.router, prefix=v1_prefix)
        app.include_router(assembly.router, prefix=v1_prefix)
        app.include_router(v1_ws.router, prefix=v1_prefix)
        app.include_router(mesh_data.router, prefix=v1_prefix)

        # Initialize v1 engine service on startup
        @app.on_event("startup")
        async def _start_v1_engine() -> None:
            try:
                get_engine_service()
            except Exception:
                logger.warning("v1 engine service initialization failed")

        @app.on_event("shutdown")
        async def _stop_v1_engine() -> None:
            try:
                shutdown_engine_service()
            except Exception:
                pass

        logger.info("v1 API routers mounted at %s", v1_prefix)
    except ImportError:
        logger.info("v1 web module not available; skipping legacy API mount")

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
