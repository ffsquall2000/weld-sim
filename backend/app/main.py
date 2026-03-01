"""FastAPI application factory for the Ultrasonic Metal Welding Virtual Simulation Platform."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from starlette.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.config import settings
from backend.app.routers import (
    acoustic as v2_acoustic,
    assembly as v2_assembly,
    calculation as v2_calculation,
    comparisons,
    geometries,
    geometry_analysis,
    health,
    horn,
    knurl as v2_knurl,
    manual_optimization,
    materials,
    mesh_data as v2_mesh_data,
    optimizations,
    projects,
    quality as quality_router,
    recipes as v2_recipes,
    reports,
    runs,
    simulations,
    suggestions as v2_suggestions,
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

    # Auto-create tables (CREATE IF NOT EXISTS)
    from backend.app.dependencies import engine
    from backend.app.models import Base  # noqa: F811

    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created / verified")
    except Exception as exc:
        logger.warning("Database table creation failed: %s", exc)

    yield
    # --- Shutdown ---
    logger.info("WeldSim v2 shutting down ...")
    await engine.dispose()


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

    # ---- GZip compression ----
    app.add_middleware(GZipMiddleware, minimum_size=500)

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
    app.include_router(horn.router, prefix=api_prefix)
    app.include_router(v2_knurl.router, prefix=api_prefix)
    app.include_router(v2_suggestions.router, prefix=api_prefix)
    app.include_router(v2_acoustic.router, prefix=api_prefix)
    app.include_router(v2_assembly.router, prefix=api_prefix)
    app.include_router(geometry_analysis.router, prefix=api_prefix)
    app.include_router(v2_mesh_data.router, prefix=api_prefix)
    app.include_router(v2_calculation.router, prefix=api_prefix)
    app.include_router(v2_recipes.router, prefix=api_prefix)
    app.include_router(quality_router.router, prefix=api_prefix)

    # ---- Initialize engine service (used by calculation, recipes, suggestions) ----
    try:
        from web.dependencies import get_engine_service, shutdown_engine_service

        @app.on_event("startup")
        async def _start_engine() -> None:
            try:
                get_engine_service()
                logger.info("Engine service initialized")
            except Exception:
                logger.warning("Engine service initialization failed")

        @app.on_event("shutdown")
        async def _stop_engine() -> None:
            try:
                shutdown_engine_service()
            except Exception:
                pass

    except ImportError:
        logger.info("Engine service not available; calculation/recipes endpoints will return 503")

    # ---- V1 API backward-compatible redirects ----
    @app.api_route(
        "/api/v1/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        include_in_schema=False,
    )
    async def v1_redirect(path: str, request: Request):
        """Redirect V1 API calls to V2 with 308 Permanent Redirect."""
        query = f"?{request.query_params}" if request.query_params else ""
        return RedirectResponse(
            url=f"/api/v2/{path}{query}",
            status_code=308,
        )

    @app.get("/docs", include_in_schema=False)
    async def old_docs_redirect():
        """Redirect old /docs URL to new /api/v2/docs."""
        return RedirectResponse(url="/api/v2/docs", status_code=301)

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
