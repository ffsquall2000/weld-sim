"""V2 acoustic analysis API endpoints.

Wraps the V1 FEAService acoustic analysis methods with Pydantic v2 schemas
and the V2 router pattern.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from backend.app.schemas.acoustic import (
    AcousticAnalysisRequest,
    AcousticAnalysisResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/acoustic", tags=["acoustic"])

# ---------------------------------------------------------------------------
# Lazy-init FEA service (tolerate missing V1 dependencies)
# ---------------------------------------------------------------------------

try:
    from web.services.fea_service import FEAService

    _fea_service = FEAService()
except ImportError:
    _fea_service = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/analyze", response_model=AcousticAnalysisResponse)
async def run_acoustic_analysis(request: AcousticAnalysisRequest):
    """Run comprehensive acoustic analysis on a horn geometry.

    Supports two solver pipelines:
    - **Gmsh TET10 + SolverA** (default, ``use_gmsh=True``): higher-accuracy
      quadratic tetrahedral mesh.
    - **Legacy HEX8** (``use_gmsh=False``): structured hexahedral mesh,
      deprecated but retained for backwards compatibility.
    """
    if _fea_service is None:
        raise HTTPException(
            status_code=503,
            detail="FEA service unavailable (missing dependencies)",
        )

    try:
        if request.use_gmsh:
            result = _fea_service.run_acoustic_analysis_gmsh(
                horn_type=request.horn_type,
                diameter_mm=request.width_mm,
                length_mm=request.height_mm,
                material=request.material,
                frequency_khz=request.frequency_khz,
                mesh_density=request.mesh_density,
            )
        else:
            result = _fea_service.run_acoustic_analysis(
                horn_type=request.horn_type,
                width_mm=request.width_mm,
                height_mm=request.height_mm,
                length_mm=request.length_mm,
                material=request.material,
                frequency_khz=request.frequency_khz,
                mesh_density=request.mesh_density,
            )
        return AcousticAnalysisResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Acoustic analysis error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Acoustic analysis failed: {exc}",
        ) from exc
