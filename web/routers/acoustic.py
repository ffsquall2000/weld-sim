"""Acoustic analysis endpoints."""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/acoustic", tags=["acoustic"])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class AcousticAnalysisRequest(BaseModel):
    horn_type: str = "cylindrical"
    width_mm: float = Field(default=25.0, gt=0)
    height_mm: float = Field(default=80.0, gt=0)
    length_mm: float = Field(default=25.0, gt=0)
    material: str = "Titanium Ti-6Al-4V"
    frequency_khz: float = Field(default=20.0, gt=0)
    mesh_density: str = "medium"


class StressHotspot(BaseModel):
    location: list[float]  # [x, y, z] in mm
    von_mises_mpa: float
    node_index: int


class AcousticAnalysisResponse(BaseModel):
    # Modal analysis
    modes: list[dict]
    closest_mode_hz: float
    target_frequency_hz: float
    frequency_deviation_percent: float
    # Harmonic response
    harmonic_response: dict  # {frequencies_hz: [...], amplitudes: [...]}
    # Amplitude distribution
    amplitude_distribution: dict  # {node_positions: [[x,y,z],...], amplitudes: [...]}
    amplitude_uniformity: float  # 0-1, higher is better
    # Stress
    stress_hotspots: list[StressHotspot]
    stress_max_mpa: float
    # Mesh info
    node_count: int
    element_count: int
    solve_time_s: float
    mesh: Optional[dict] = None


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post("/analyze", response_model=AcousticAnalysisResponse)
async def run_acoustic_analysis(request: AcousticAnalysisRequest):
    """Run comprehensive acoustic analysis on a horn geometry."""
    try:
        from web.services.fea_service import FEAService

        svc = FEAService()
        result = svc.run_acoustic_analysis(
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
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        logger.error("Acoustic analysis error: %s", exc, exc_info=True)
        raise HTTPException(500, f"Acoustic analysis failed: {exc}") from exc
