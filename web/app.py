"""FastAPI application factory."""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from web.config import WebConfig
from web.dependencies import get_engine_service, shutdown_engine_service
from web.routers import health, calculation, materials, recipes, reports, geometry
from web.routers import horn, acoustic, knurl, knurl_fea, suggestions, assembly
from web.routers import ws, mesh_data

# Configure application logging so logger.info() calls in web.* modules
# actually produce output.  Uvicorn only configures its own loggers.
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s: %(message)s",
)
# Silence noisy third-party loggers
logging.getLogger("multipart.multipart").setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up the engine on startup; shut it down on stop."""
    get_engine_service()
    yield
    shutdown_engine_service()


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    application = FastAPI(
        title="UltrasonicWeldMaster API",
        version="0.1.0",
        lifespan=lifespan,
    )

    # GZip compression for all responses > 500 bytes
    application.add_middleware(GZipMiddleware, minimum_size=500)

    # CORS
    application.add_middleware(
        CORSMiddleware,
        allow_origins=WebConfig.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    application.include_router(health.router, prefix="/api/v1")
    application.include_router(calculation.router, prefix="/api/v1")
    application.include_router(materials.router, prefix="/api/v1")
    application.include_router(recipes.router, prefix="/api/v1")
    application.include_router(reports.router, prefix="/api/v1")
    application.include_router(geometry.router, prefix="/api/v1")
    application.include_router(horn.router, prefix="/api/v1")
    application.include_router(acoustic.router, prefix="/api/v1")
    application.include_router(knurl.router, prefix="/api/v1")
    application.include_router(knurl_fea.router, prefix="/api/v1")
    application.include_router(suggestions.router, prefix="/api/v1")
    application.include_router(assembly.router, prefix="/api/v1")
    application.include_router(ws.router, prefix="/api/v1")
    application.include_router(mesh_data.router, prefix="/api/v1")

    # Serve frontend static files if the build directory exists
    frontend_dist = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
    if os.path.isdir(frontend_dist):
        index_html = os.path.join(frontend_dist, "index.html")

        # Static assets (JS, CSS, images, etc.)
        application.mount(
            "/assets",
            StaticFiles(directory=os.path.join(frontend_dist, "assets")),
            name="assets",
        )

        # Serve specific static files at root level
        @application.get("/vite.svg", include_in_schema=False)
        async def vite_svg():
            return FileResponse(os.path.join(frontend_dist, "vite.svg"))

        # SPA fallback: any non-API, non-asset path returns index.html
        # so Vue Router handles client-side routing (e.g. /geometry, /calculate)
        @application.get("/{full_path:path}", include_in_schema=False)
        async def spa_fallback(request: Request, full_path: str):
            # Don't intercept API or WebSocket paths
            if full_path.startswith("api/") or full_path.startswith("ws/"):
                from fastapi.responses import JSONResponse
                return JSONResponse({"detail": "Not Found"}, status_code=404)
            return FileResponse(index_html)

    return application


app = create_app()
