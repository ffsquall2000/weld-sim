"""Horn generation and chamfer analysis endpoints."""
from __future__ import annotations

import logging
import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from web.schemas.horn import (
    HornGenerateRequest,
    HornGenerateResponse,
    ChamferAnalysisRequest,
    ChamferAnalysisResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/horn", tags=["horn"])

# Singleton service instance
_horn_service = None


def _get_horn_service():
    global _horn_service
    if _horn_service is None:
        from web.services.horn_service import HornService

        _horn_service = HornService()
    return _horn_service


@router.post("/generate", response_model=HornGenerateResponse)
async def generate_horn(request: HornGenerateRequest):
    """Generate a parametric horn with knurl and chamfer."""
    try:
        from ultrasonic_weld_master.plugins.geometry_analyzer.horn_generator import (
            HornParams,
        )

        params = HornParams(
            horn_type=request.horn_type,
            width_mm=request.width_mm,
            height_mm=request.height_mm,
            length_mm=request.length_mm,
            material=request.material,
            knurl_type=request.knurl_type,
            knurl_pitch_mm=request.knurl_pitch_mm,
            knurl_tooth_width_mm=request.knurl_tooth_width_mm,
            knurl_depth_mm=request.knurl_depth_mm,
            knurl_direction=request.knurl_direction,
            chamfer_radius_mm=request.chamfer_radius_mm,
            chamfer_angle_deg=request.chamfer_angle_deg,
            edge_treatment=request.edge_treatment or "none",
        )

        svc = _get_horn_service()
        result, download_id = svc.generate_horn(params)

        return HornGenerateResponse(
            horn_type=result.horn_type,
            dimensions=result.dimensions,
            knurl_info=result.knurl_info,
            chamfer_info=result.chamfer_info,
            volume_mm3=round(result.volume_mm3, 2),
            surface_area_mm2=round(result.surface_area_mm2, 2),
            has_cad_export=result.has_cad_export,
            mesh=result.mesh,
            download_id=download_id,
        )
    except Exception as exc:
        logger.error("Horn generation error: %s", exc, exc_info=True)
        raise HTTPException(500, f"Horn generation failed: {exc}") from exc


@router.post("/chamfer-analysis", response_model=ChamferAnalysisResponse)
async def analyze_chamfer(request: ChamferAnalysisRequest):
    """Analyze the impact of chamfer/edge treatment on welding."""
    try:
        svc = _get_horn_service()
        result = svc.analyze_chamfer(
            contact_width_mm=request.contact_width_mm,
            contact_length_mm=request.contact_length_mm,
            chamfer_radius_mm=request.chamfer_radius_mm,
            chamfer_angle_deg=request.chamfer_angle_deg,
            edge_treatment=request.edge_treatment or "none",
            pressure_mpa=request.pressure_mpa,
            amplitude_um=request.amplitude_um,
            material_yield_mpa=request.material_yield_mpa,
            weld_time_s=request.weld_time_s,
        )
        return ChamferAnalysisResponse(**result)
    except Exception as exc:
        logger.error("Chamfer analysis error: %s", exc, exc_info=True)
        raise HTTPException(
            500, f"Chamfer analysis failed: {exc}"
        ) from exc


@router.get("/download/{file_id}")
async def download_horn_file(file_id: str, fmt: str = "step"):
    """Download generated horn file (STEP or STL)."""
    if fmt not in ("step", "stl"):
        raise HTTPException(400, "Format must be 'step' or 'stl'")

    svc = _get_horn_service()
    path = svc.get_download_path(file_id, fmt)
    if path is None:
        raise HTTPException(404, "File not found or expired")

    media_type = (
        "application/step" if fmt == "step" else "application/sla"
    )
    filename = f"horn_{file_id}.{fmt}"
    return FileResponse(path, media_type=media_type, filename=filename)
