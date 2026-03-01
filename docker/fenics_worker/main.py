"""FEniCSx solver worker -- FastAPI application.

This service exposes a REST API that the main ultrasonic-weld-master
application (port 8001) can call to offload heavy FEA computations
to a DOLFINx/PETSc/SLEPc backend.

Endpoint stubs return HTTP 501 until Task 24 (SolverB) fills them in.
"""
from __future__ import annotations

import os
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="FEniCSx Solver Worker",
    description="REST API for FEA computations using DOLFINx (SolverB backend)",
    version="0.1.0",
)

_START_TIME = time.time()

# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class ModalRequest(BaseModel):
    """Payload for a modal (eigenvalue) analysis request."""
    mesh_file: str = Field(..., description="Path to mesh file inside /data/meshes")
    material_name: str = Field(..., description="Material identifier")
    n_modes: int = Field(default=20, ge=1, le=200)
    target_frequency_hz: float = Field(default=20000.0, gt=0)
    boundary_conditions: str = Field(default="free-free")


class HarmonicRequest(BaseModel):
    """Payload for a harmonic response analysis request."""
    mesh_file: str = Field(..., description="Path to mesh file inside /data/meshes")
    material_name: str = Field(..., description="Material identifier")
    freq_min_hz: float = Field(default=16000.0, gt=0)
    freq_max_hz: float = Field(default=24000.0, gt=0)
    n_freq_points: int = Field(default=201, ge=2)
    damping_model: str = Field(default="hysteretic")
    damping_ratio: float = Field(default=0.005, ge=0)
    excitation_node_set: str = Field(default="bottom_face")
    response_node_set: str = Field(default="top_face")


class StaticRequest(BaseModel):
    """Payload for a static stress analysis request."""
    mesh_file: str = Field(..., description="Path to mesh file inside /data/meshes")
    material_name: str = Field(..., description="Material identifier")
    loads: list[dict[str, Any]] = Field(default_factory=list)
    boundary_conditions: list[dict[str, Any]] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Health-check response."""
    status: str
    uptime_s: float
    version: str
    solver_backend: str


class AnalysisResponse(BaseModel):
    """Generic analysis response wrapper."""
    status: str
    message: str
    result: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health-check endpoint for Docker and load-balancer probes."""
    return HealthResponse(
        status="ok",
        uptime_s=round(time.time() - _START_TIME, 2),
        version="0.1.0",
        solver_backend="dolfinx-0.8.0",
    )


@app.post("/api/v1/modal", response_model=AnalysisResponse)
async def modal_analysis(request: ModalRequest) -> AnalysisResponse:
    """Run modal (eigenvalue) analysis via FEniCSx.

    .. note:: Stub -- returns 501 until SolverB (Task 24) is implemented.
    """
    raise HTTPException(
        status_code=501,
        detail="Modal analysis not yet implemented. Awaiting SolverB (Task 24).",
    )


@app.post("/api/v1/harmonic", response_model=AnalysisResponse)
async def harmonic_analysis(request: HarmonicRequest) -> AnalysisResponse:
    """Run harmonic response analysis via FEniCSx.

    .. note:: Stub -- returns 501 until SolverB (Task 24) is implemented.
    """
    raise HTTPException(
        status_code=501,
        detail="Harmonic analysis not yet implemented. Awaiting SolverB (Task 24).",
    )


@app.post("/api/v1/static", response_model=AnalysisResponse)
async def static_analysis(request: StaticRequest) -> AnalysisResponse:
    """Run static stress analysis via FEniCSx.

    .. note:: Stub -- returns 501 until SolverB (Task 24) is implemented.
    """
    raise HTTPException(
        status_code=501,
        detail="Static analysis not yet implemented. Awaiting SolverB (Task 24).",
    )
